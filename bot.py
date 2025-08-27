# bot.py — Nature Inspire (Clarity + HDR) и Nature Inspire 2.0 (HDR only) + ESRGAN
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
MODEL_CLARITY = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"
MODEL_ESRGAN  = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"

# ---------- TUNABLES (крутилки) ----------
# Общие ограничения
INPUT_MAX_SIDE        = 1536                    # ресайз перед моделями Replicate
FINAL_TELEGRAM_LIMIT  = 10 * 1024 * 1024        # 10MB

# Clarity
CLARITY_SCALE_FACTOR     = 2
CLARITY_DYNAMIC          = 6.0
CLARITY_CREATIVITY       = 0.25
CLARITY_RESEMBLANCE      = 0.65
CLARITY_TILING_W         = 112
CLARITY_TILING_H         = 144
CLARITY_STEPS            = 22
CLARITY_SD_MODEL         = "juggernaut_reborn.safetensors [338b85bc4f]"
CLARITY_SCHEDULER        = "DPM++ 3M SDE Karras"
CLARITY_MORE_DETAILS_LORA= 0.5                   # <lora:more_details:x>
CLARITY_RENDER_LORA      = 1.0                   # <lora:SDXLrender_v2.0:x>

# HDR сила (0..1)
HDR_STRENGTH_LOW   = 0.35
HDR_STRENGTH_MED   = 0.60
HDR_STRENGTH_HIGH  = 0.85

# Доп. «ручки» внутри HDR
HDR_EXPOSURE_BASE  = 1.06                        # глобальная экспозиция (1.00..1.40)
HDR_EXPOSURE_GAIN  = 0.30                        # вклад от strength в экспозицию
HDR_LOG_A_BASE     = 2.0                         # параметр лог-томапа (2..6)
HDR_LOG_A_GAIN     = 3.0

# ESRGAN
UPSCALE_AFTER_HDR        = True                  # финализировать ESRGAN
UPSCALE_SCALE            = 2                     # 2 или 4
ESRGAN_MAX_INPUT_PIXELS  = 2_000_000             # безопасный лимит входа для ESRGAN

# ---------- STATE ----------
# user_id -> {'effect': ..., 'strength': float}
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

# ---------- HDR (лог-тонмап, с анти-«серостью») ----------
def _pil_gaussian(img: Image.Image, radius: float) -> Image.Image:
    small = img.resize((max(8, img.width//2), max(8, img.height//2)), Image.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(radius=radius*0.75))
    return small.resize(img.size, Image.LANCZOS)

def hdr_enhance_path(orig_path: str, strength: float = 0.6) -> str:
    """
    Натуральный HDR без «пластика»:
      1) глобальная экспозиция (поднимаем midtones),
      2) лог-тонмап на яркости (компресс хайлайтов, подъём теней),
      3) локальный контраст мягко, немного насыщенности,
      4) авто-компенсация яркости, если стало темнее.
    """
    im = Image.open(orig_path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    arr = np.asarray(im).astype(np.float32) / 255.0

    # базовая средняя яркость для контроля
    in_luma = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    in_mean = float(in_luma.mean())

    # 1) глобальная экспозиция
    exposure = HDR_EXPOSURE_BASE + HDR_EXPOSURE_GAIN * strength   # ~1.16..1.31
    arr = np.clip(arr * exposure, 0.0, 1.0)

    # 2) лог-тонмап по луме с сохранением цвета
    luma = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]
    a = HDR_LOG_A_BASE + HDR_LOG_A_GAIN * strength                 # 2..5.5
    y_new = np.log1p(a * luma) / (np.log1p(a) + 1e-8)
    ratio = y_new / np.maximum(luma, 1e-6)
    arr *= ratio[..., None]
    arr = np.clip(arr, 0.0, 1.0)

    base = Image.fromarray((arr*255).astype(np.uint8))

    # мягкие маски для донастройки теней/хайлайтов
    l = np.asarray(base.convert("L")).astype(np.float32) / 255.0
    sh = np.clip(1.0 - l*1.1, 0.0, 1.0)
    hl = np.clip((l - 0.75)*2.0, 0.0, 1.0)
    sh_mask = np.asarray(_pil_gaussian(Image.fromarray((sh*255).astype(np.uint8)), 3.0), dtype=np.float32)/255.0
    hl_mask = np.asarray(_pil_gaussian(Image.fromarray((hl*255).astype(np.uint8)), 3.0), dtype=np.float32)/255.0

    arr2 = np.asarray(base).astype(np.float32) / 255.0
    sh_gain = 0.12 + 0.22*strength
    hl_cut  = 0.06 + 0.12*strength
    for c in range(3):
        chan = arr2[...,c]
        chan = chan + sh_mask * sh_gain * (1.0 - chan)
        chan = chan - hl_mask * hl_cut * chan
        arr2[...,c] = np.clip(chan, 0.0, 1.0)
    base = Image.fromarray((arr2*255).astype(np.uint8))

    # локальный контраст + лёгкая насыщенность
    blurred = base.filter(ImageFilter.GaussianBlur(radius=1.3 + 2.8*strength))
    hp = ImageChops.subtract(base, blurred)
    hp = hp.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(110+100*strength), threshold=3))
    mc_amount = 0.15 + 0.22*strength
    base = Image.blend(base, hp, mc_amount)

    base = base.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80+int(80*strength), threshold=2))
    base = ImageEnhance.Color(base).enhance(1.06 + 0.16*strength)

    # 4) если стало темнее — компенсируем
    out_l = np.asarray(base.convert("L")).astype(np.float32)/255.0
    out_mean = float(out_l.mean())
    if out_mean < in_mean * 0.98:
        gain = min(1.40, max(1.00, (in_mean / max(out_mean, 1e-6)) ** 0.85))
        base = ImageEnhance.Brightness(base).enhance(gain)

    fd, out_path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    base.save(out_path, "JPEG", quality=95, optimize=True)
    return out_path

# ---------- ESRGAN (безопасный вход под лимит GPU) ----------
def esrgan_upscale_path(path: str, scale: int = 2) -> str:
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    w, h = im.size
    px = w * h
    if px > ESRGAN_MAX_INPUT_PIXELS:
        k = (ESRGAN_MAX_INPUT_PIXELS / px) ** 0.5
        nw, nh = max(256, int(w * k)), max(256, int(h * k))
        im = im.resize((nw, nh), Image.LANCZOS)
        fd, safe_in = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        im.save(safe_in, "JPEG", quality=95, optimize=True)
        in_path = safe_in
    else:
        in_path = path

    with open(in_path, "rb") as bf:
        out = replicate.run(MODEL_ESRGAN, input={"image": bf, "scale": scale})
    url = pick_first_url(out)
    tmp = download_to_temp(url)

    if in_path != path:
        try: os.remove(in_path)
        except: pass
    return tmp

# ---------- PIPELINES ----------
async def run_nature_enhance_clarity_hdr(file_id: str, strength: float) -> str:
    # 1) качаем фото и ресайзим перед Replicate
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)

    # 2) Clarity по файлу (контроль размера)
    prompt_text = (
        "masterpiece, best quality, highres,\n"
        f"<lora:more_details:{CLARITY_MORE_DETAILS_LORA}>\n"
        f"<lora:SDXLrender_v2.0:{CLARITY_RENDER_LORA}>"
    )
    negative = "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"

    with open(local_in, "rb") as f:
        cl_out = replicate.run(
            MODEL_CLARITY,
            input={
                "image": f,
                "prompt": prompt_text,
                "negative_prompt": negative,
                "scale_factor": CLARITY_SCALE_FACTOR,
                "dynamic": CLARITY_DYNAMIC,
                "creativity": CLARITY_CREATIVITY,
                "resemblance": CLARITY_RESEMBLANCE,
                "tiling_width": CLARITY_TILING_W,
                "tiling_height": CLARITY_TILING_H,
                "sd_model": CLARITY_SD_MODEL,
                "scheduler": CLARITY_SCHEDULER,
                "num_inference_steps": CLARITY_STEPS,
                "seed": 1337,
                "downscaling": False,
                "sharpen": 0,
                "handfix": "disabled",
                "output_format": "png",
            }
        )
    try: os.remove(local_in)
    except: pass

    cl_url  = pick_first_url(cl_out)
    cl_path = download_to_temp(cl_url)

    # 3) HDR
    try:
        hdr_path = hdr_enhance_path(cl_path, strength=strength)
    finally:
        try: os.remove(cl_path)
        except: pass

    # 4) ESRGAN по флажку
    if UPSCALE_AFTER_HDR:
        up_path = esrgan_upscale_path(hdr_path, scale=UPSCALE_SCALE)
        try: os.remove(hdr_path)
        except: pass
        hdr_path = up_path

    return hdr_path

async def run_nature_enhance_hdr_only(file_id: str, strength: float) -> str:
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)
    hdr_path = hdr_enhance_path(local_in, strength=strength)
    try: os.remove(local_in)
    except: pass

    if UPSCALE_AFTER_HDR:
        up_path = esrgan_upscale_path(hdr_path, scale=UPSCALE_SCALE)
        try: os.remove(hdr_path)
        except: pass
        hdr_path = up_path

    return hdr_path

# ---------- UI ----------
KB_MAIN = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance (Clarity + HDR)")],
        [KeyboardButton("🌿 Nature Enhance 2.0 (HDR only)")],
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
        "• Nature Enhance — Clarity + натуральный HDR (+ESRGAN финал)\n"
        "• Nature Enhance 2.0 — только натуральный HDR (+ESRGAN финал)\n"
        "Выбери режим и силу эффекта.",
        reply_markup=KB_MAIN
    )

@dp.message_handler(lambda m: m.text in [
    "🌿 Nature Enhance (Clarity + HDR)",
    "🌿 Nature Enhance 2.0 (HDR only)"
])
async def on_choose_mode(m: types.Message):
    uid = m.from_user.id
    WAIT[uid] = {"effect": "clarity_menu" if "Clarity" in m.text else "hdr_menu"}
    await m.answer("Выбери силу эффекта:", reply_markup=KB_STRENGTH)

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

    strength = HDR_STRENGTH_MED
    if m.text == "Низкая":  strength = HDR_STRENGTH_LOW
    if m.text == "Высокая": strength = HDR_STRENGTH_HIGH

    if st["effect"] == "clarity_menu":
        WAIT[uid] = {"effect": "clarity_hdr", "strength": strength}
        await m.answer("Пришли фото — сделаю Nature Enhance (Clarity + HDR) 🌿", reply_markup=KB_MAIN)
    elif st["effect"] == "hdr_menu":
        WAIT[uid] = {"effect": "hdr_only", "strength": strength}
        await m.answer("Пришли фото — сделаю Nature Enhance 2.0 (HDR only) 🌿", reply_markup=KB_MAIN)

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st or st.get("effect") not in ["clarity_hdr", "hdr_only"]:
        await m.reply("Сначала выбери режим и силу эффекта ⬇️", reply_markup=KB_MAIN)
        return

    await m.reply("⏳ Обрабатываю...")
    try:
        strength = float(st.get("strength", HDR_STRENGTH_MED))
        if st["effect"] == "clarity_hdr":
            out_path = await run_nature_enhance_clarity_hdr(m.photo[-1].file_id, strength=strength)
        else:
            out_path = await run_nature_enhance_hdr_only(m.photo[-1].file_id, strength=strength)

        safe_path = ensure_size_under_telegram_limit(out_path)
        await m.reply_photo(InputFile(safe_path))
        try:
            if os.path.exists(out_path): os.remove(out_path)
            if safe_path != out_path and os.path.exists(safe_path): os.remove(safe_path)
        except: pass

    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка Nature Inspire:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling…")
    executor.start_polling(dp, skip_updates=True)
