# bot.py — Nature Inspire (фикс микса): HDR-only = Nature Enhance 2.0, WOW = сочный топ-пайплайн
# + 🎻 Violin Touch (твои значения сохранены)
# + Анти-флэр (точечное подавление голубых бликов) после любого эффекта
# env: TELEGRAM_API_TOKEN, REPLICATE_API_TOKEN (опц., для Clarity)

import os, logging, tempfile, urllib.request, traceback
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InputFile
from aiogram.utils import executor

# --- опц. для Clarity ---
try:
    import replicate
except Exception:
    replicate = None

logging.basicConfig(level=logging.INFO)

# ---------- TOGGLES ----------
ANTI_FLARE_ON = True   # можно выключить, поставив False

# ---------- TOKENS ----------
API_TOKEN  = os.getenv("TELEGRAM_API_TOKEN")
if not API_TOKEN:  raise RuntimeError("TELEGRAM_API_TOKEN missing")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")  # может быть None
if REPL_TOKEN and replicate:
    os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN

bot = Bot(token=API_TOKEN)
dp  = Dispatcher(bot)

# ---------- TUNABLES ----------
INPUT_MAX_SIDE       = 1536
FINAL_TELEGRAM_LIMIT = 10 * 1024 * 1024

# UI уровни (мягкий множитель для всех компонент; сами компоненты крутятся отдельно ниже)
UI_LOW, UI_MED, UI_HIGH = 0.01, 0.50, 1.00

# ==== WOW: РАЗДЕЛЬНЫЕ КРУТИЛКИ ==================================================
# 1) COLOR — насыщенность/вибранс (деликатно поднимает «плоские» цвета)
COLOR_VIBRANCE_BASE   = 0.42
COLOR_CONTRAST_BASE   = 0.15
COLOR_BRIGHT_BASE     = 0.02

# 2) DEPTH — «объём»: S-кривая, микроконтраст (high-pass), финальный шарп
DEPTH_S_CURVE_BASE    = 0.35
DEPTH_MICROCONTR_BASE = 0.2
DEPTH_HP_RADIUS_BASE  = 1.40
DEPTH_UNSHARP_BASE    = 130

# 3) DRAMA — драматизм: HDR-кривая (лог), bloom хайлайтов
DRAMA_HDR_LOGA_BASE   = 3
DRAMA_BLOOM_AMOUNT    = 0.9
DRAMA_BLOOM_RADIUS    = 2.00

# Анти-серость (гарантия, что не потемнеет)
ANTI_GREY_TOL = 0.98
ANTI_GREY_CAP = 1.35

# --- Clarity (Replicate) ---
MODEL_CLARITY = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"
CL_SCALE_FACTOR      = 2
CL_DYNAMIC           = 6.5
CL_CREATIVITY        = 0.25
CL_RESEMBLANCE       = 0.70
CL_TILING_W, CL_TILING_H = 112, 144
CL_STEPS             = 18
CL_SD_MODEL          = "juggernaut_reborn.safetensors [338b85bc4f]"
CL_SCHEDULER         = "DPM++ 3M SDE Karras"
CL_NEGATIVE          = "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"
CL_LORA_MORE_DETAILS = 0.52
CL_LORA_RENDER       = 1.0
# ================================================================================

# ---------- STATE ----------
# user_id -> {'effect': 'ne2' | 'wow_menu' | 'wow' | 'violin_menu' | 'violin' | 'violin_boost', 'ui_gain': float}
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

def download_to_temp(url: str, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix); os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def _pick_first_url(x):
    try:
        if isinstance(x, str): return x
        if isinstance(x, (list, tuple)) and x:
            o0 = x[0]; u = getattr(o0, "url", None)
            return u() if callable(u) else (u or str(o0))
        u = getattr(x, "url", None)
        return u() if callable(u) else (u or str(x))
    except:
        return str(x)

def _anti_flare_blue(im_pil, hi_thr=0.82, blue_h1=150, blue_h2=210, desat=0.60, warm=0.02):
    """
    Гасит сине-циановые ореолы на очень ярких бликах:
    - ищем хайлайты по луме,
    - внутри хайлайтов приглушаем насыщенность только для синего диапазона,
    - чуть «согреваем» (тёплый tint), чтобы вода/солнце не уходили в холод.
    """
    arr = np.asarray(im_pil).astype(np.float32) / 255.0
    # лума
    y = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    hi = (y >= hi_thr).astype(np.float32)

    # RGB -> HSV
    hsv = Image.fromarray((arr*255).astype(np.uint8)).convert("HSV")
    hsv = np.asarray(hsv).astype(np.float32)
    H, S, V = hsv[...,0], hsv[...,1]/255.0, hsv[...,2]/255.0

    # маска синего в хайлайтах
    blue = (((H >= blue_h1) & (H <= blue_h2)).astype(np.float32)) * hi
    if blue.max() > 0:
        S2 = S*(1.0 - desat*blue)                 # десатурируем только синий в бликах
        V2 = np.clip(V + warm*blue, 0.0, 1.0)     # лёгкий «тёплый» сдвиг
        hsv[...,1] = (S2*255.0)
        hsv[...,2] = (V2*255.0)

    out = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    return out

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
    out = ImageEnhance.Brightness(out).enhance(1.00)
    out = ImageEnhance.Contrast(out).enhance(1.06)

    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    out.save(path, "JPEG", quality=95, optimize=True)
    return path

def wow_enhance_path(orig_path: str, ui_gain: float) -> str:
    """
    WOW-пайплайн: сочный топ.
    ui_gain — мягкий множитель кнопки (0.1 / 0.50 / 1.00).
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
    🎻 Violin Touch — «музыкальный» цвет/объём (чуть темнее и сочнее).
    Без внешних моделей; только PIL/NumPy.
    >>> ВАЖНО: значения оставлены как ты настроил.
    """
    base = Image.open(orig_path).convert("RGB")
    base = ImageOps.exif_transpose(base)
    arr  = np.asarray(base).astype(np.float32)/255.0

    # 1) HDR-лог мягче (чтобы не высветлять)
    l = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    A = 2.7
    y = np.log1p(A*l) / (np.log1p(A)+1e-8)
    arr = np.clip(arr * (y/np.maximum(l,1e-6))[...,None], 0, 1)

    # 2) Плёночная S-кривая (чуть сильнее для объёма)
    arr = _s_curve(arr, amt=0.24)

    # 3) Vibrance с защитой кожи
    im_hsv = Image.fromarray((arr*255).astype(np.uint8)).convert("HSV")
    hsv    = np.asarray(im_hsv).astype(np.float32)
    H,S,V  = hsv[...,0], hsv[...,1], hsv[...,2]
    skin = (((H>=15) & (H<=35)) & (S>20) & (V>40)).astype(np.float32)

    def _vib(a, gain):
        mx = a.max(axis=-1, keepdims=True); mn = a.min(axis=-1, keepdims=True)
        sat = mx - mn; w = 1.0 - sat; mean = a.mean(axis=-1, keepdims=True)
        return np.clip(mean + (a-mean)*(1.0 + gain*w), 0.0, 1.0)

    vib_gain = 0.48
    vib = _vib(arr, vib_gain)
    arr = np.clip(arr*(skin[...,None]) + vib*(1.0-skin[...,None]), 0, 1)

    im = Image.fromarray((arr*255).astype(np.uint8))

    # 4) Локальный контраст + лёгкий bloom
    hp = ImageChops.subtract(im, im.filter(ImageFilter.GaussianBlur(radius=1.2)))
    im = Image.blend(im, hp, 0.32)
    glow = im.filter(ImageFilter.GaussianBlur(radius=2.0))
    im = Image.blend(im, ImageChops.screen(im, glow), 0.04)

    # 4.1) Анти-флэр для сине-циановых бликов
    im = _anti_flare_blue(im, hi_thr=0.82, blue_h1=150, blue_h2=210, desat=0.60, warm=0.02)

    # 5) Общие правки: цвет/контраст, без осветления
    im = ImageEnhance.Color(im).enhance(1.08)
    im = ImageEnhance.Contrast(im).enhance(1.14)
    im = ImageEnhance.Brightness(im).enhance(1.00)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    im.save(path, "JPEG", quality=95, optimize=True)
    return path

# ---------- АНТИ-ФЛЭР ----------
def anti_flare_pass(im: Image.Image) -> Image.Image:
    """
    Точечно гасит голубые/циановые всполохи в хайлайтах (линз-флэр).
    Логика: HSV-маска hue∈[170..210]°, sat>60, value>140 (uint8), размытие маски,
    затем для маски: снижаем S, слегка снижаем V, hue тянем к тёплому (~30°).
    """
    if not ANTI_FLARE_ON:
        return im

    hsv = im.convert("HSV")
    arr = np.asarray(hsv).astype(np.float32)  # H,S,V ∈ [0..255]
    H, S, V = arr[...,0], arr[...,1], arr[...,2]

    # маска «циан/синий в хайлайтах»
    mask = (
        ((H >= 170) & (H <= 210)) &
        (S >= 60) &
        (V >= 140)
    ).astype(np.uint8)*255

    # размягчить границы
    m_img = Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(radius=6))
    m = np.asarray(m_img).astype(np.float32)/255.0  # [0..1]

    if m.max() < 0.02:  # почти нет синих бликов — выходим
        return im

    # применить коррекцию по маске
    warm_hue = 30.0  # тёплый тон
    S_corr = S * (1.0 - 0.70*m)     # сильная десатурация блика
    V_corr = V * (1.0 - 0.15*m)     # чуть темнее, чтобы убрать «светлячок»
    H_corr = H*(1.0 - 0.60*m) + warm_hue*(0.60*m)

    arr[...,0] = np.clip(H_corr, 0, 255)
    arr[...,1] = np.clip(S_corr, 0, 255)
    arr[...,2] = np.clip(V_corr, 0, 255)

    out = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")
    return out

def anti_flare_path(in_path: str) -> str:
    """Открыть -> anti_flare_pass -> сохранить как новый jpg, вернуть путь."""
    try:
        im = Image.open(in_path).convert("RGB")
        out = anti_flare_pass(im)
        fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        out.save(path, "JPEG", quality=95, optimize=True)
        return path
    except Exception:
        return in_path

# ---------- CLARITY (мягкий пост-проход) ----------
def clarity_post_path(local_path: str) -> str:
    """Нежный Clarity как финальный штрих. Если токена/репликейта нет — вернём исходник."""
    if not (REPL_TOKEN and replicate):
        return local_path
    try:
        with open(local_path, "rb") as f:
            out = replicate.run(MODEL_CLARITY, input={
                "image": f,
                "prompt": "<lora:more_details:%s>\n<lora:SDXLrender_v2.0:%s>" % (CL_LORA_MORE_DETAILS, CL_LORA_RENDER),
                "negative_prompt": CL_NEGATIVE,
                "scale_factor": CL_SCALE_FACTOR,
                "dynamic": CL_DYNAMIC,
                "creativity": CL_CREATIVITY,
                "resemblance": CL_RESEMBLANCE,
                "tiling_width": CL_TILING_W,
                "tiling_height": CL_TILING_H,
                "sd_model": CL_SD_MODEL,
                "scheduler": CL_SCHEDULER,
                "num_inference_steps": CL_STEPS,
                "downscaling": False,
                "sharpen": 0,
                "handfix": "disabled",
                "output_format": "png",
                "seed": 1337
            })
        url = _pick_first_url(out)
        if not url:
            return local_path
        new_path = download_to_temp(url, ".png")
        im = Image.open(new_path).convert("RGB")
        fd, jpg = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        im.save(jpg, "JPEG", quality=95, optimize=True)
        try: os.remove(new_path)
        except: pass
        return jpg
    except Exception:
        return local_path

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
KB_VIOLIN = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("⬅️ Назад")],
        [KeyboardButton("Обычный 🎻"), KeyboardButton("Усиление 🎻")],
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

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance 2.0 (HDR)", "🌿 WOW Enhance (в разработке)", "🎻 Violin Touch", "Обычный 🎻", "Усиление 🎻"])
async def on_mode(m: types.Message):
    uid = m.from_user.id
    txt = m.text
    if txt == "🌿 WOW Enhance (в разработке)":
        WAIT[uid] = {"effect": "wow_menu"}
        await m.answer("Выбери силу эффекта:", reply_markup=KB_STRENGTH)
    elif txt == "🎻 Violin Touch":
        WAIT[uid] = {"effect": "violin_menu"}
        await m.answer("Выбери вариант 🎻:", reply_markup=KB_VIOLIN)
    elif txt == "Обычный 🎻":
        WAIT[uid] = {"effect": "violin"}
        await m.answer("Пришли фото — сделаю 🎻 Violin Touch", reply_markup=KB_MAIN)
    elif txt == "Усиление 🎻":
        WAIT[uid] = {"effect": "violin_boost"}
        await m.answer("Пришли фото — сделаю 🎻 Violin Touch (усиление)", reply_markup=KB_MAIN)
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
    if st.get("effect") != "wow_menu":
        return
    ui_gain = UI_MED
    if m.text == "Низкая":  ui_gain = UI_LOW
    if m.text == "Высокая": ui_gain = UI_HIGH
    WAIT[uid] = {"effect": "wow", "ui_gain": float(ui_gain)}
    await m.answer("Пришли фото — сделаю WOW Enhance 🌿", reply_markup=KB_MAIN)

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st  = WAIT.get(uid)
    if not st or st.get("effect") not in ["ne2", "wow", "violin", "violin_boost"]:
        await m.reply("Сначала выбери режим ⬇️", reply_markup=KB_MAIN); return

    await m.reply("⏳ Обрабатываю...")
    try:
        local = await download_tg_photo(m.photo[-1].file_id)

        eff = st["effect"]
        if eff == "ne2":
            out = hdr_only_path(local)

        elif eff == "violin":
            out = violin_touch_path(local)

        elif eff == "violin_boost":
            tmp = violin_touch_path(local)
            tmp2 = clarity_post_path(tmp)  # мягкий clarity-штрих
            out  = anti_flare_path(tmp2) if ANTI_FLARE_ON else tmp2
            try:
                for p in [tmp, tmp2]:
                    if p != out and os.path.exists(p): os.remove(p)
            except: pass

        else:  # wow
            tmp = wow_enhance_path(local, ui_gain=float(st.get("ui_gain", UI_MED)))
            tmp2 = clarity_post_path(tmp)  # мягкий clarity-штрих
            out  = anti_flare_path(tmp2) if ANTI_FLARE_ON else tmp2
            try:
                for p in [tmp, tmp2]:
                    if p != out and os.path.exists(p): os.remove(p)
            except: pass

        # анти-флэр для NE2/обычного Violin тоже применим (без Clarity)
        if eff in ["ne2", "violin"] and ANTI_FLARE_ON:
            new_out = anti_flare_path(out)
            if new_out != out:
                try:
                    if os.path.exists(out): os.remove(out)
                except: pass
                out = new_out

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
