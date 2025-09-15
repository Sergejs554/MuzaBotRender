# bot.py — Nature Inspire (фикс микса): HDR-only = Nature Enhance 2.0, WOW = сочный топ-пайплайн
# + 🎻 Violin Touch (подправлен под «глубокий» вид неба/воды)
# env: TELEGRAM_API_TOKEN

import os, logging, tempfile, urllib.request, traceback
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InputFile
from aiogram.utils import executor

logging.basicConfig(level=logging.INFO)

# ---------- TOKENS ----------
API_TOKEN  = os.getenv("TELEGRAM_API_TOKEN")
if not API_TOKEN:  raise RuntimeError("TELEGRAM_API_TOKEN missing")
bot = Bot(token=API_TOKEN)
dp  = Dispatcher(bot)

# ---------- TUNABLES ----------
INPUT_MAX_SIDE       = 1536
FINAL_TELEGRAM_LIMIT = 10 * 1024 * 1024

# UI уровни (мягкий множитель для всех компонент; сами компоненты крутятся отдельно ниже)
UI_LOW, UI_MED, UI_HIGH = 0.85, 1.00, 1.15

# ==== WOW: РАЗДЕЛЬНЫЕ КРУТИЛКИ ==================================================
# 1) COLOR — насыщенность/вибранс (деликатно поднимает «плоские» цвета)
COLOR_VIBRANCE_BASE   = 0.36
COLOR_CONTRAST_BASE   = 0.12
COLOR_BRIGHT_BASE     = 0.06

# 2) DEPTH — «объём»: S-кривая, микроконтраст (high-pass), финальный шарп
DEPTH_S_CURVE_BASE    = 0.22
DEPTH_MICROCONTR_BASE = 0.30
DEPTH_HP_RADIUS_BASE  = 1.40
DEPTH_UNSHARP_BASE    = 130

# 3) DRAMA — драматизм: HDR-кривая (лог), bloom хайлайтов
DRAMA_HDR_LOGA_BASE   = 3.9
DRAMA_BLOOM_AMOUNT    = 0.18
DRAMA_BLOOM_RADIUS    = 2.00

# Анти-серость (гарантия, что не потемнеет)
ANTI_GREY_TOL = 0.98
ANTI_GREY_CAP = 1.35
# ================================================================================

# ---------- STATE ----------
# user_id -> {'effect': 'ne2' | 'wow_menu' | 'wow' | 'violin', 'ui_gain': float}
WAIT = {}

# ---------- HELPERS ----------
def resize_inplace(path: str, max_side: int):
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(path, "JPEG", quality=95, optimize=True)
    except Exception:
        pass

def ensure_size_under_telegram_limit(path: str, max_bytes: int = FINAL_TELEGRAM_LIMIT) -> str:
    try:
        if os.path.getsize(path) <= max_bytes: return path
        img = Image.open(path).convert("RGB")
        q = 92
        for _ in range(10):
            fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
            img.save(tmp, "JPEG", quality=q, optimize=True)
            if os.path.getsize(tmp) <= max_bytes:
                os.remove(path); return tmp
            os.remove(tmp); q -= 8
        fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        img.save(tmp, "JPEG", quality=max(q, 40), optimize=True)
        os.remove(path); return tmp
    except Exception:
        return path

def tg_url(file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

async def download_tg_photo(file_id: str) -> str:
    tg_file = await bot.get_file(file_id)
    url = tg_url(tg_file.file_path)
    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    urllib.request.urlretrieve(url, path)
    resize_inplace(path, INPUT_MAX_SIDE)
    return path

# ---------- CORE OPS ----------
def _vibrance(arr: np.ndarray, gain: float) -> np.ndarray:
    mx = arr.max(axis=-1, keepdims=True)
    mn = arr.min(axis=-1, keepdims=True)
    sat = mx - mn
    w   = 1.0 - sat          # больше буст там, где мало насыщенности
    mean = arr.mean(axis=-1, keepdims=True)
    out  = mean + (arr - mean) * (1.0 + gain * w)
    return np.clip(out, 0.0, 1.0)

def _s_curve(x: np.ndarray, amt: float) -> np.ndarray:
    return np.clip(x*(1-amt) + (3*x*x - 2*x*x*x)*amt, 0.0, 1.0)

# ---------- EFFECTS ----------
def hdr_only_path(orig_path: str) -> str:
    """Натуральный HDR-only для Nature Enhance 2.0 (без серости)."""
    im = Image.open(orig_path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    a = 3.0
    arr = np.asarray(im).astype(np.float32)/255.0
    luma = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    y    = np.log1p(a*luma) / (np.log1p(a)+1e-8)
    ratio = y / np.maximum(luma, 1e-6)
    arr = np.clip(arr * ratio[...,None], 0.0, 1.0)

    out = Image.fromarray((arr*255).astype(np.uint8))
    out = ImageEnhance.Brightness(out).enhance(1.04)
    out = ImageEnhance.Contrast(out).enhance(1.06)

    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    out.save(path, "JPEG", quality=95, optimize=True)
    return path

def wow_enhance_path(orig_path: str, ui_gain: float) -> str:
    """
    WOW-пайплайн: сочный топ.
    ui_gain — мягкий множитель кнопки (0.85 / 1.00 / 1.15).
    """
    g = float(ui_gain)

    base = Image.open(orig_path).convert("RGB")
    base = ImageOps.exif_transpose(base)
    arr  = np.asarray(base).astype(np.float32)/255.0

    # яркость для анти-серости
    in_luma = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    in_mean = float(in_luma.mean())

    # DRAMA: HDR (лог по луме)
    A = DRAMA_HDR_LOGA_BASE * g
    y = np.log1p(A*in_luma) / (np.log1p(A)+1e-8)
    ratio = y / np.maximum(in_luma, 1e-6)
    arr = np.clip(arr * ratio[...,None], 0.0, 1.0)

    # DEPTH: S-curve
    arr = _s_curve(arr, amt= DEPTH_S_CURVE_BASE * g)

    # COLOR: Vibrance
    arr = _vibrance(arr, gain= COLOR_VIBRANCE_BASE * g)

    # COLOR глобальные
    im = Image.fromarray((arr*255).astype(np.uint8))
    im = ImageEnhance.Contrast(im).enhance(1.0 + COLOR_CONTRAST_BASE * g)
    im = ImageEnhance.Brightness(im).enhance(1.0 + COLOR_BRIGHT_BASE  * g)

    # DEPTH: Microcontrast (high-pass)
    hp_r = DEPTH_HP_RADIUS_BASE * g
    blurred = im.filter(ImageFilter.GaussianBlur(radius=hp_r))
    hp = ImageChops.subtract(im, blurred)
    hp = hp.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(90 + 110*g), threshold=3))
    im = Image.blend(im, hp, min(0.6, DEPTH_MICROCONTR_BASE * g))

    # DRAMA: Bloom хайлайтов
    if DRAMA_BLOOM_AMOUNT > 0:
        glow = im.filter(ImageFilter.GaussianBlur(radius=DRAMA_BLOOM_RADIUS + 4.0*g))
        im = Image.blend(im, ImageChops.screen(im, glow), DRAMA_BLOOM_AMOUNT * g)

    # DEPTH: финальный микрошарп
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(DEPTH_UNSHARP_BASE * g), threshold=2))

    # анти-серость
    out_mean = np.asarray(im.convert("L")).astype(np.float32).mean()/255.0
    if out_mean < in_mean * ANTI_GREY_TOL:
        gain = min(ANTI_GREY_CAP, max(1.00, (in_mean / max(out_mean, 1e-6)) ** 0.85))
        im = ImageEnhance.Brightness(im).enhance(gain)

    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    im.save(path, "JPEG", quality=95, optimize=True)
    return path
    
def violin_touch_path(orig_path: str) -> str:
    """
    🎻 Violin Touch — мягкий «музыкальный» цвет/объём без артефактов.
    Float-пайплайн + лёгкий dither против постеризации.
    """
    # --- загрузка в float ---
    base = Image.open(orig_path).convert("RGB")
    base = ImageOps.exif_transpose(base)
    arr  = np.asarray(base).astype(np.float32) / 255.0

    # --- 1) HDR-лог (чуть мягче) ---
    l = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    A = 3.2  # было 3.6
    y = np.log1p(A*l) / (np.log1p(A) + 1e-8)
    arr = np.clip(arr * (y/np.maximum(l, 1e-6))[..., None], 0.0, 1.0)

    # --- 2) S-curve (чуть мягче) ---
    def _s_curve_np(x, amt):
        return np.clip(x*(1-amt) + (3*x*x - 2*x*x*x)*amt, 0.0, 1.0)
    arr = _s_curve_np(arr, amt=0.16)  # было 0.20

    # --- 3) Vibrance с защитой кожи (маска по HSV только для маски) ---
    pil_for_mask = Image.fromarray((arr*255).astype(np.uint8)).convert("HSV")
    hsv = np.asarray(pil_for_mask).astype(np.float32)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    skin = (((H>=15) & (H<=35)) & (S>20) & (V>40)).astype(np.float32)
    # vibrance в float
    def _vibrance_np(a, gain):
        mx = a.max(axis=-1, keepdims=True); mn = a.min(axis=-1, keepdims=True)
        sat = mx - mn; mean = a.mean(axis=-1, keepdims=True)
        w = 1.0 - sat
        return np.clip(mean + (a-mean)*(1.0 + gain*w), 0.0, 1.0)
    vib_gain = 0.24  # было 0.32
    vib = _vibrance_np(arr, vib_gain)
    arr = arr*skin[...,None] + vib*(1.0 - skin[...,None])

    # --- 4) Локальный контраст (high-pass) в float + мягкий bloom ---
    def _gauss(im_arr, r):
        pil = Image.fromarray((np.clip(im_arr,0,1)*255).astype(np.uint8))
        blr = pil.filter(ImageFilter.GaussianBlur(radius=r))
        return np.asarray(blr).astype(np.float32)/255.0

    hp_blur = _gauss(arr, 1.0)                      # было 1.2
    hp = np.clip(arr - hp_blur, -1.0, 1.0)
    arr = np.clip(arr + 0.18*hp, 0.0, 1.0)          # было 0.28

    glow = _gauss(arr, 1.8)                         # было 2.4
    screen = 1.0 - (1.0 - arr)*(1.0 - glow)
    arr = np.clip(arr*(1-0.08) + screen*0.08, 0.0, 1.0)  # было 0.10

    # --- 5) Общие правки (контраст/яркость) ---
    arr = np.clip(arr*1.00, 0.0, 1.0)
    pil = Image.fromarray((arr*255).astype(np.uint8))
    pil = ImageEnhance.Contrast(pil).enhance(1.06)  # было 1.10
    pil = ImageEnhance.Brightness(pil).enhance(1.02)  # было 1.03

    # назад в float для финального микрошарпа и dither
    arr = np.asarray(pil).astype(np.float32)/255.0

    # лёгкий микрошарп в float
    sh_blur = _gauss(arr, 0.6)
    arr = np.clip(arr + (arr - sh_blur)*0.15, 0.0, 1.0)

    # --- DITHER: микрошум против «полос/квадратов» ---
    rng = np.random.default_rng()
    arr += rng.uniform(-1/255*0.6, 1/255*0.6, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    # --- сохранение без хрома-сабсемплинга ---
    out = Image.fromarray((arr*255).astype(np.uint8))
    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    out.save(path, "JPEG", quality=95, optimize=True, subsampling=0)
    return path

# ---------- UI ----------
KB_MAIN = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance 2.0 (HDR)")],
        [KeyboardButton("🌿 WOW Enhance (в разработке)")],
        [KeyboardButton("🎻 Violin Touch")],
    ],
    resize_keyboard=True
)
KB_STRENGTH = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("⬅️ Назад")],
        [KeyboardButton("Низкая"), KeyboardButton("Средняя"), KeyboardButton("Высокая")],
    ], resize_keyboard=True
)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer(
        "Nature Inspire 🌿\n"
        "• Nature Enhance 2.0 — HDR-only (мягкий, без серости)\n"
        "• WOW Enhance — сочный топ-пайплайн (в разработке)\n"
        "• 🎻 Violin Touch — музыкальный цвет/объём (глубокие синие)\n"
        "Выбери режим.",
        reply_markup=KB_MAIN
    )

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance 2.0 (HDR)", "🌿 WOW Enhance (в разработке)", "🎻 Violin Touch"])
async def on_mode(m: types.Message):
    uid = m.from_user.id
    if "WOW" in m.text:
        WAIT[uid] = {"effect": "wow_menu"}
        await m.answer("Выбери силу эффекта:", reply_markup=KB_STRENGTH)
    elif "Violin" in m.text:
        WAIT[uid] = {"effect": "violin"}
        await m.answer("Пришли фото — сделаю 🎻 Violin Touch", reply_markup=KB_MAIN)
    else:
        WAIT[uid] = {"effect": "ne2"}
        await m.answer("Пришли фото — сделаю Nature Enhance 2.0 🌿", reply_markup=KB_MAIN)

@dp.message_handler(lambda m: m.text in ["Низкая","Средняя","Высокая","⬅️ Назад"])
async def on_strength(m: types.Message):
    uid = m.from_user.id
    st  = WAIT.get(uid)
    if not st: return
    if m.text == "⬅️ Назад":
        WAIT.pop(uid, None); await m.answer("Главное меню.", reply_markup=KB_MAIN); return
    ui_gain = UI_MED
    if m.text == "Низкая":  ui_gain = UI_LOW
    if m.text == "Высокая": ui_gain = UI_HIGH
    WAIT[uid] = {"effect": "wow", "ui_gain": float(ui_gain)}
    await m.answer("Пришли фото — сделаю WOW Enhance 🌿", reply_markup=KB_MAIN)

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st  = WAIT.get(uid)
    if not st or st.get("effect") not in ["ne2", "wow", "violin"]:
        await m.reply("Сначала выбери режим ⬇️", reply_markup=KB_MAIN); return

    await m.reply("⏳ Обрабатываю...")
    try:
        local = await download_tg_photo(m.photo[-1].file_id)

        eff = st["effect"]
        if eff == "ne2":
            out = hdr_only_path(local)
        elif eff == "violin":
            out = violin_touch_path(local)
        else:
            out = wow_enhance_path(local, ui_gain=float(st.get("ui_gain", UI_MED)))

        safe = ensure_size_under_telegram_limit(out)
        await m.reply_photo(InputFile(safe))
        try:
            for p in [local, out]:
                if os.path.exists(p): os.remove(p)
            if safe != out and os.path.exists(safe): os.remove(safe)
        except: pass
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка Nature Inspire:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling…")
    executor.start_polling(dp, skip_updates=True)
