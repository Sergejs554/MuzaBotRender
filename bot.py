# bot.py — Nature Inspire: (2.0) Clarity-only и WOW Enhance с крутилкой
# env: TELEGRAM_API_TOKEN, REPLICATE_API_TOKEN

import os, logging, tempfile, urllib.request, traceback
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops
import replicate
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InputFile
from aiogram.utils import executor

logging.basicConfig(level=logging.INFO)

# ---------- TOKENS ----------
API_TOKEN  = os.getenv("TELEGRAM_API_TOKEN")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("TELEGRAM_API_TOKEN missing")
if not REPL_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN missing")
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ---------- MODELS ----------
# Nature Enhance 2.0 = Clarity Upscaler (как было)
MODEL_CLARITY = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"

# ---------- TUNABLES ----------
INPUT_MAX_SIDE       = 1536                 # ресайз перед моделями Replicate
FINAL_TELEGRAM_LIMIT = 10 * 1024 * 1024     # 10MB

# Clarity (как в твоём исходном коде, бережные)
CL_SCALE_FACTOR      = 2
CL_DYNAMIC           = 6.0
CL_CREATIVITY        = 0.25
CL_RESEMBLANCE       = 0.65
CL_TILING_W          = 112
CL_TILING_H          = 144
CL_STEPS             = 22
CL_SD_MODEL          = "juggernaut_reborn.safetensors [338b85bc4f]"
CL_SCHEDULER         = "DPM++ 3M SDE Karras"
CL_LORA_MORE_DETAILS = 0.50
CL_LORA_RENDER       = 1.00
CL_NEGATIVE          = "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"

# WOW — уровни силы (кнопки: Низкая/Средняя/Высокая)
WOW_LEVEL_LOW    = 1.3
WOW_LEVEL_MED    = 1.6
WOW_LEVEL_HIGH   = 1.9   # просил 1.1 — сделал так

# База «вау»-эффекта (можешь менять вручную)
WOW_BASE = {
    "vibrance_gain":   0.18,   # сколько добавляем «вибранса» (щадящая насыщенность)
    "contrast_gain":   0.10,   # глобальный контраст
    "brightness_gain": 0.04,   # общий свет
    "curve_amount":    0.18,   # S-кривая (плёночная)
    "log_a":           2.8,    # лог-тонмап (HDR) — чем выше, тем сильнее
    "microcontrast":   0.22,   # локальный контраст (high-pass blend)
    "blur_radius":     1.6,    # базовый радиус гаусса для high-pass
    "unsharp_percent": 110     # финальный Unsharp
}

# ---------- STATE ----------
# user_id -> {'effect': 'ne2'|'wow_menu'|'wow', 'strength': float}
WAIT = {}

# ---------- HELPERS ----------
def tg_public_url(file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

async def download_tg_photo(file_id: str) -> str:
    tg_file = await bot.get_file(file_id)
    url = tg_public_url(tg_file.file_path)
    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def resize_inplace(path: str, max_side: int):
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        img.save(path, "JPEG", quality=95, optimize=True)
    except Exception:
        pass

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def ensure_size_under_telegram_limit(path: str, max_bytes: int = FINAL_TELEGRAM_LIMIT) -> str:
    try:
        if os.path.getsize(path) <= max_bytes:
            return path
        img = Image.open(path).convert("RGB")
        q = 92
        for _ in range(10):
            fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
            img.save(tmp, "JPEG", quality=q, optimize=True)
            if os.path.getsize(tmp) <= max_bytes:
                try: os.remove(path)
                except: pass
                return tmp
            os.remove(tmp); q -= 8
        fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        img.save(tmp, "JPEG", quality=max(q, 40), optimize=True)
        try: os.remove(path)
        except: pass
        return tmp
    except Exception:
        return path

def pick_first_url(output) -> str:
    try:
        if isinstance(output, str):
            return output
        if isinstance(output, (list, tuple)) and output:
            o0 = output[0]
            url_attr = getattr(o0, "url", None)
            return url_attr() if callable(url_attr) else (url_attr or str(o0))
        url_attr = getattr(output, "url", None)
        return url_attr() if callable(url_attr) else (url_attr or str(output))
    except Exception:
        return str(output)

# ---------- WOW PIPELINE (локальный) ----------
def _vibrance(img_arr: np.ndarray, gain: float) -> np.ndarray:
    # «вибранс»: усиливаем низконасыщенные области сильнее
    mx = img_arr.max(axis=-1, keepdims=True)
    mn = img_arr.min(axis=-1, keepdims=True)
    sat = mx - mn                               # 0..1
    w = 1.0 - sat                               # серые области получают больший буст
    mean = img_arr.mean(axis=-1, keepdims=True)
    # масштабируем от центра, чтобы не уводить баланс
    boost = 1.0 + gain * w
    out = mean + (img_arr - mean) * boost
    return np.clip(out, 0.0, 1.0)

def _s_curve(x: np.ndarray, amt: float) -> np.ndarray:
    # плавная S-кривая: mix линейного и smoothstep
    y = x*(1-amt) + (3*x*x - 2*x*x*x)*amt
    return np.clip(y, 0.0, 1.0)

def wow_enhance_path(orig_path: str, strength: float) -> str:
    """
    WOW: натуральный «вау» без пластика. Всё масштабируется параметром strength.
    """
    s = float(strength)
    base = Image.open(orig_path).convert("RGB")
    base = ImageOps.exif_transpose(base)

    # в numpy
    arr = np.asarray(base).astype(np.float32) / 255.0

    # 1) лёгкий HDR-тонмап (лог по луме)
    l = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    a = max(1.0, WOW_BASE["log_a"] * s)
    y = np.log1p(a*l) / (np.log1p(a) + 1e-8)
    ratio = y / np.maximum(l, 1e-6)
    arr = np.clip(arr * ratio[...,None], 0.0, 1.0)

    # 2) S-кривая (киношная глубина)
    arr = _s_curve(arr, amt= WOW_BASE["curve_amount"] * s)

    # 3) Vibrance (щадящая насыщенность)
    arr = _vibrance(arr, gain= WOW_BASE["vibrance_gain"] * s)

    # 4) Контраст/яркость
    arr = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((arr*255).astype(np.uint8))
    im = ImageEnhance.Contrast(im).enhance(1.0 + WOW_BASE["contrast_gain"] * s)
    im = ImageEnhance.Brightness(im).enhance(1.0 + WOW_BASE["brightness_gain"] * s)

    # 5) Локальный «кларити» (high-pass)
    blur_r = WOW_BASE["blur_radius"] + 2.2*s
    blurred = im.filter(ImageFilter.GaussianBlur(radius=blur_r))
    hp = ImageChops.subtract(im, blurred)
    hp = hp.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(90 + 80*s), threshold=3))
    im = Image.blend(im, hp, min(0.5, WOW_BASE["microcontrast"] * s))

    # 6) Финальный микрошарп
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(WOW_BASE["unsharp_percent"] * s), threshold=2))

    fd, out_path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    im.save(out_path, "JPEG", quality=95, optimize=True)
    return out_path

# ---------- PIPELINES ----------
async def run_nature_enhance_v2_clarity_only(file_id: str) -> str:
    """
    Nature Enhance 2.0 — как было: CLARITY UPSCALER с LoRA, без доп. шагов.
    """
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)

    prompt_text = (
        "masterpiece, best quality, highres,\n"
        f"<lora:more_details:{CL_LORA_MORE_DETAILS}>\n"
        f"<lora:SDXLrender_v2.0:{CL_LORA_RENDER}>"
    )
    try:
        with open(local_in, "rb") as f:
            out = replicate.run(
                MODEL_CLARITY,
                input={
                    "image": f,
                    "prompt": prompt_text,
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
                    "seed": 1337,
                    "downscaling": False,
                    "sharpen": 0,
                    "handfix": "disabled",
                    "output_format": "png",
                }
            )
        url = pick_first_url(out)
        out_path = download_to_temp(url)
    finally:
        try: os.remove(local_in)
        except: pass

    return out_path

async def run_wow(file_id: str, strength: float) -> str:
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)
    try:
        out_path = wow_enhance_path(local_in, strength=strength)
    finally:
        try: os.remove(local_in)
        except: pass
    return out_path

# ---------- UI ----------
KB_MAIN = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance 2.0")],
        [KeyboardButton("🌿 WOW Enhance")],
    ],
    resize_keyboard=True
)

KB_STRENGTH = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("⬅️ Назад")],
        [KeyboardButton("Низкая"), KeyboardButton("Средняя"), KeyboardButton("Высокая")],
    ],
    resize_keyboard=True
)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer(
        "Nature Inspire 🌿\n"
        "• Nature Enhance 2.0 — Clarity Upscaler (как было)\n"
        "• WOW Enhance — сочность+глубина (с крутилкой силы)\n"
        "Выбери режим.",
        reply_markup=KB_MAIN
    )

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance 2.0", "🌿 WOW Enhance"])
async def on_choose_mode(m: types.Message):
    uid = m.from_user.id
    if "WOW" in m.text:
        WAIT[uid] = {"effect": "wow_menu"}
        await m.answer("Выбери силу эффекта:", reply_markup=KB_STRENGTH)
    else:
        WAIT[uid] = {"effect": "ne2"}
        await m.answer("Пришли фото — сделаю Nature Enhance 2.0 🌿", reply_markup=KB_MAIN)

@dp.message_handler(lambda m: m.text in ["Низкая", "Средняя", "Высокая", "⬅️ Назад"])
async def on_strength(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st:
        return
    if m.text == "⬅️ Назад":
        WAIT.pop(uid, None)
        await m.answer("Главное меню.", reply_markup=KB_MAIN)
        return

    strength = WOW_LEVEL_MED
    if m.text == "Низкая":  strength = WOW_LEVEL_LOW
    if m.text == "Высокая": strength = WOW_LEVEL_HIGH

    WAIT[uid] = {"effect": "wow", "strength": float(strength)}
    await m.answer("Пришли фото — сделаю WOW Enhance 🌿", reply_markup=KB_MAIN)

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st or st.get("effect") not in ["ne2", "wow"]:
        await m.reply("Сначала выбери режим ⬇️", reply_markup=KB_MAIN)
        return

    await m.reply("⏳ Обрабатываю...")
    try:
        if st["effect"] == "ne2":
            out_path = await run_nature_enhance_v2_clarity_only(m.photo[-1].file_id)
        else:
            out_path = await run_wow(m.photo[-1].file_id, strength=float(st.get("strength", WOW_LEVEL_MED)))

        safe = ensure_size_under_telegram_limit(out_path)
        await m.reply_photo(InputFile(safe))
        try:
            if os.path.exists(out_path): os.remove(out_path)
            if safe != out_path and os.path.exists(safe): os.remove(safe)
        except: pass
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка Nature Inspire:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling…")
    executor.start_polling(dp, skip_updates=True)
