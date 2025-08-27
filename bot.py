# bot.py — Nature Inspire: Clarity-only (2.0) и WOW Enhance (топ-пайплайн + крутилка)
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
INPUT_MAX_SIDE       = 1536                 # ресайз входа перед моделями Replicate
FINAL_TELEGRAM_LIMIT = 10 * 1024 * 1024     # 10MB

# Clarity (бережные, как в твоих исходниках)
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

# WOW — уровни силы (кнопки)
WOW_LEVEL_LOW    = 0.60
WOW_LEVEL_MED    = 0.80
WOW_LEVEL_HIGH   = 1.10   # как просил (1.1)

# База «вау»-эффекта (топ-пайплайн). Все множители масштабируются strength.
WOW_BASE = {
    # 1) HDR тонмап (лог по луме)
    "log_a":            3.4,    # сила HDR-кривой (мягкая компрессия хайлайтов, подъём теней)
    # 2) Плёночная S-кривая (глубина)
    "curve_amount":     0.22,
    # 3) Vibrance (щадящая насыщенность: больше там, где мало цвета)
    "vibrance_gain":    0.22,
    # 4) Глобальные
    "contrast_gain":    0.12,
    "brightness_gain":  0.06,
    # 5) Microcontrast (high-pass blend)
    "microcontrast":    0.30,   # доля примеси HP
    "hp_blur_base":     1.4,    # базовый радиус blur перед high-pass
    # 6) Bloom (хайлайты, сияние)
    "bloom_amount":     0.12,
    "bloom_radius":     2.0,    # базовый радиус
    # 7) Финальный микрошарп
    "unsharp_percent":  130,
}

# Анти-серость: если средняя яркость после тонмапа падает >2% — деликатно компенсируем
ANTI_GREY_TOL = 0.98
ANTI_GREY_CAP = 1.35

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

# ---------- WOW PIPELINE (ТОП) ----------
def _vibrance(arr: np.ndarray, gain: float) -> np.ndarray:
    # Усиливаем низконасыщенные области сильнее, высоко-насыщенные — деликатно
    mx = arr.max(axis=-1, keepdims=True)
    mn = arr.min(axis=-1, keepdims=True)
    sat = mx - mn
    w = 1.0 - sat
    mean = arr.mean(axis=-1, keepdims=True)
    boost = 1.0 + gain * w
    out = mean + (arr - mean) * boost
    return np.clip(out, 0.0, 1.0)

def _s_curve(x: np.ndarray, amt: float) -> np.ndarray:
    return np.clip(x*(1-amt) + (3*x*x - 2*x*x*x)*amt, 0.0, 1.0)

def wow_enhance_path(orig_path: str, strength: float) -> str:
    s = float(strength)
    base = Image.open(orig_path).convert("RGB")
    base = ImageOps.exif_transpose(base)
    arr = np.asarray(base).astype(np.float32) / 255.0

    # Базовая яркость для анти-серости
    in_luma = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    in_mean = float(in_luma.mean())

    # 1) HDR (лог-тонмап по луме)
    a = max(1.0, WOW_BASE["log_a"] * s)
    y  = np.log1p(a*in_luma) / (np.log1p(a) + 1e-8)
    ratio = y / np.maximum(in_luma, 1e-6)
    arr = np.clip(arr * ratio[...,None], 0.0, 1.0)

    # 2) Плёночная S-кривая
    arr = _s_curve(arr, amt= WOW_BASE["curve_amount"] * s)

    # 3) Vibrance
    arr = _vibrance(arr, gain= WOW_BASE["vibrance_gain"] * s)

    # 4) Контраст / Яркость
    im = Image.fromarray((arr*255).astype(np.uint8))
    im = ImageEnhance.Contrast(im).enhance(1.0 + WOW_BASE["contrast_gain"] * s)
    im = ImageEnhance.Brightness(im).enhance(1.0 + WOW_BASE["brightness_gain"] * s)

    # 5) Microcontrast (high-pass)
    hp_r = WOW_BASE["hp_blur_base"] + 2.0*s
    blurred = im.filter(ImageFilter.GaussianBlur(radius=hp_r))
    hp = ImageChops.subtract(im, blurred)
    hp = hp.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(100 + 110*s), threshold=3))
    im = Image.blend(im, hp, min(0.6, WOW_BASE["microcontrast"] * s))

    # 6) Bloom (хайлайты)
    if WOW_BASE["bloom_amount"] > 0:
        glow_r = WOW_BASE["bloom_radius"] + 5.0*s
        glow = im.filter(ImageFilter.GaussianBlur(radius=glow_r))
        im = Image.blend(im, ImageChops.screen(im, glow), WOW_BASE["bloom_amount"] * s)

    # 7) Финальный микрошарп
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(WOW_BASE["unsharp_percent"] * s), threshold=2))

    # Анти-серость: если итог темнее исходного, слегка компенсируем
    out_mean = np.asarray(im.convert("L")).astype(np.float32).mean() / 255.0
    if out_mean < in_mean * ANTI_GREY_TOL:
        gain = min(ANTI_GREY_CAP, max(1.00, (in_mean / max(out_mean, 1e-6)) ** 0.85))
        im = ImageEnhance.Brightness(im).enhance(gain)

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
        "• WOW Enhance — HDR+Vibrance+Depth+Bloom (крутилка силы)\n"
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
