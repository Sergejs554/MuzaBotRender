# bot.py — Nature Inspire (2 кнопки: Nature Enhance / Nature Enhance 2.0)
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
MODEL_CLARITY  = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"
MODEL_REFINER  = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_ESRGAN   = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
MODEL_SWIN2SR  = "mv-lab/swin2sr:a01b0512004918ca55d02e554914a9eca63909fa83a29ff0f115c78a7045574f"  # fallback

# ---------- TUNABLES ----------
INPUT_MAX_SIDE           = 1536                  # ресайз перед моделями (stability)
FINAL_TELEGRAM_LIMIT     = 10 * 1024 * 1024      # 10MB

# Clarity (бережные настройки)
CL_SCALE_FACTOR          = 2
CL_DYNAMIC               = 5.0
CL_CREATIVITY            = 0.22
CL_RESEMBLANCE           = 0.72
CL_TILING_W              = 112
CL_TILING_H              = 144
CL_STEPS                 = 20
CL_SD_MODEL              = "juggernaut_reborn.safetensors [338b85bc4f]"
CL_SCHEDULER             = "DPM++ 3M SDE Karras"
CL_LORA_MORE_DETAILS     = 0.45
CL_LORA_RENDER           = 0.9

# Refiner prompt (натуральное усиление, без пластика)
REFINER_PROMPT = (
    "enhance photo clarity, natural detail, preserve realistic colors, "
    "no plastic skin, DSLR-like rendering, avoid over-sharpening"
)

# Upscale
UPSCALE_AFTER            = False                  # по умолчанию выкл. (чтобы не ловить OOM)
UPSCALE_ENGINE           = "swin2sr"              # 'esrgan' | 'swin2sr'
UPSCALE_SCALE            = 2                      # только для ESRGAN
ESRGAN_MAX_INPUT_PIXELS  = 1_400_000              # агрессивнее лимит для ESRGAN, чтобы избегать OOM
ESRGAN_RETRIES           = 3

# ---------- STATE ----------
WAIT = {}  # user_id -> {'effect': 'ne'|'ne2'}

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

# ---------- UPSCALE ENGINES ----------
def _run_swin2sr(path: str) -> str:
    with open(path, "rb") as bf:
        out = replicate.run(MODEL_SWIN2SR, input={"image": bf})
    url = pick_first_url(out)
    return download_to_temp(url)

def _run_esrgan_safe(path: str, scale: int = 2,
                     max_pixels: int = ESRGAN_MAX_INPUT_PIXELS,
                     retries: int = ESRGAN_RETRIES) -> str:
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)

    def _save_resized(img, tgt_pixels):
        w, h = img.size
        px = w * h
        if px > tgt_pixels:
            k = (tgt_pixels / px) ** 0.5
            nw, nh = max(256, int(w * k)), max(256, int(h * k))
            img = img.resize((nw, nh), Image.LANCZOS)
        fd, tmp_in = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        img.save(tmp_in, "JPEG", quality=95, optimize=True)
        return tmp_in

    attempt = 0
    cur_pixels = max_pixels
    last_tmp = None
    while attempt <= retries:
        if last_tmp and os.path.exists(last_tmp):
            try: os.remove(last_tmp)
            except: pass
        tmp_in = _save_resized(im, cur_pixels)
        last_tmp = tmp_in
        try:
            with open(tmp_in, "rb") as bf:
                out = replicate.run(MODEL_ESRGAN, input={"image": bf, "scale": scale})
            url = pick_first_url(out)
            out_path = download_to_temp(url)
            try: os.remove(tmp_in)
            except: pass
            return out_path
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "max size" in msg or "fits in gpu memory" in msg:
                attempt += 1
                cur_pixels = int(cur_pixels * 0.7)
                continue
            break

    # fallback
    if last_tmp and os.path.exists(last_tmp):
        try: os.remove(last_tmp)
        except: pass
    return _run_swin2sr(path)

def maybe_upscale(path: str) -> str:
    if not UPSCALE_AFTER:
        return path
    if UPSCALE_ENGINE == "esrgan":
        return _run_esrgan_safe(path, scale=UPSCALE_SCALE)
    else:
        return _run_swin2sr(path)

# ---------- PIPELINES ----------
async def run_nature_enhance_basic(file_id: str) -> str:
    """
    Nature Enhance: Magic Image Refiner (натуральное «про-камера» улучшение)
    """
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)
    try:
        with open(local_in, "rb") as f:
            ref_out = replicate.run(
                MODEL_REFINER,
                input={
                    "image": f,
                    "prompt": REFINER_PROMPT,
                }
            )
        ref_url = pick_first_url(ref_out)
        ref_path = download_to_temp(ref_url)
    finally:
        try: os.remove(local_in)
        except: pass

    # (опц.) апскейл
    out_path = maybe_upscale(ref_path)
    if out_path != ref_path:
        try: os.remove(ref_path)
        except: pass
    return out_path

async def run_nature_enhance_v2(file_id: str) -> str:
    """
    Nature Enhance 2.0: Clarity (бережно) → Magic Image Refiner
    """
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)

    # 1) Clarity по файлу (контроль размера)
    prompt_text = (
        "masterpiece, best quality, highres,\n"
        f"<lora:more_details:{CL_LORA_MORE_DETAILS}>\n"
        f"<lora:SDXLrender_v2.0:{CL_LORA_RENDER}>"
    )
    negative = "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"
    try:
        with open(local_in, "rb") as f:
            cl_out = replicate.run(
                MODEL_CLARITY,
                input={
                    "image": f,
                    "prompt": prompt_text,
                    "negative_prompt": negative,
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
        cl_url  = pick_first_url(cl_out)
        cl_path = download_to_temp(cl_url)
    finally:
        try: os.remove(local_in)
        except: pass

    # 2) Refiner поверх результата
    try:
        with open(cl_path, "rb") as f:
            ref_out = replicate.run(
                MODEL_REFINER,
                input={
                    "image": f,
                    "prompt": REFINER_PROMPT,
                }
            )
        ref_url = pick_first_url(ref_out)
        ref_path = download_to_temp(ref_url)
    finally:
        try: os.remove(cl_path)
        except: pass

    # (опц.) апскейл
    out_path = maybe_upscale(ref_path)
    if out_path != ref_path:
        try: os.remove(ref_path)
        except: pass
    return out_path

# ---------- UI ----------
KB_MAIN = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance")],
        [KeyboardButton("🌿 Nature Enhance 2.0")],
    ],
    resize_keyboard=True
)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer(
        "Nature Inspire 🌿\n"
        "• Nature Enhance — Magic Image Refiner (натуральная «про-камера» обработка)\n"
        "• Nature Enhance 2.0 — Clarity → Magic Image Refiner (бережно, детальнее)\n"
        "Пришли фото после выбора режима.",
        reply_markup=KB_MAIN
    )

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance", "🌿 Nature Enhance 2.0"])
async def on_choose_mode(m: types.Message):
    uid = m.from_user.id
    WAIT[uid] = {"effect": "ne" if "2.0" not in m.text else "ne2"}
    await m.answer("Ок! Пришли фото.")

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st or st.get("effect") not in ["ne", "ne2"]:
        await m.reply("Сначала выбери режим ⬇️", reply_markup=KB_MAIN)
        return

    await m.reply("⏳ Обрабатываю...")
    try:
        if st["effect"] == "ne":
            out_path = await run_nature_enhance_basic(m.photo[-1].file_id)
        else:
            out_path = await run_nature_enhance_v2(m.photo[-1].file_id)

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
