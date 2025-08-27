# bot.py — Nature Inspire (Replicate) — CLARITY + Натуральный HDR (две версии)
# env: TELEGRAM_API_TOKEN, REPLICATE_API_TOKEN

import os, logging, tempfile, urllib.request, traceback
from io import BytesIO

# === добавлено ===
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops
# === /добавлено ===

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

# ---------- MODEL REFS ----------
MODEL_CLARITY = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"

# ---------- STATE ----------
# user_id -> {'effect': 'nature_menu'|'nature2_menu'|'nature_clarity_hdr'|'nature_hdr', 'strength': float}
WAIT = {}

# ---------- HELPERS ----------
def tg_public_url(file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

async def telegram_file_to_public_url(file_id: str) -> str:
    tg_file = await bot.get_file(file_id)
    return tg_public_url(tg_file.file_path)

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def ensure_photo_size_under_telegram_limit(path: str, max_bytes: int = 10 * 1024 * 1024) -> str:
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
            os.remove(tmp)
            q -= 8
        fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        img.save(tmp, "JPEG", quality=max(q, 40), optimize=True)
        try: os.remove(path)
        except: pass
        return tmp
    except Exception:
        return path

def pick_first_url(output) -> str:
    """
    У clarity-upscaler реплай — чаще список blob-объектов.
    Возвращаем URL первого элемента. Если пришла строка — её.
    """
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

# ---------- НАТУРАЛЬНЫЙ HDR (без генерации, без «пластика») ----------
# === добавлено ===
def _pil_gaussian(img: Image.Image, radius: float) -> Image.Image:
    # экономное размытие (down/up + GaussianBlur) для мягких масок
    small = img.resize((max(8, img.width//2), max(8, img.height//2)), Image.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(radius=radius*0.75))
    return small.resize(img.size, Image.LANCZOS)

def hdr_enhance_path(orig_path: str, strength: float = 0.6) -> str:
    """
    Натуральный HDR-тонмаппинг:
      - поднимаем тени, приглушаем хайлайты (по luma) мягкими масками,
      - локальный контраст (микро-деталь) без ореолов,
      - лёгкая насыщенность.
    strength: 0..1  (0.35 — мягко, 0.6 — средне, 0.85 — агрессивнее)
    """
    im = Image.open(orig_path).convert("RGB")
    im = ImageOps.exif_transpose(im)

    arr = np.asarray(im).astype(np.float32) / 255.0
    # luma Rec.709 приблизительно
    luma = 0.2627*arr[...,0] + 0.6780*arr[...,1] + 0.0593*arr[...,2]

    # маски для теней/хайлайтов
    shadows = np.clip(1.0 - luma*1.2, 0.0, 1.0)
    highlights = np.clip((luma - 0.65)*1.7, 0.0, 1.0)

    sh_mask_img = Image.fromarray((shadows*255).astype(np.uint8))
    hl_mask_img = Image.fromarray((highlights*255).astype(np.uint8))
    sh_mask_img = _pil_gaussian(sh_mask_img, 3.0)
    hl_mask_img = _pil_gaussian(hl_mask_img, 3.0)
    sh_mask = np.asarray(sh_mask_img, dtype=np.float32)/255.0
    hl_mask = np.asarray(hl_mask_img, dtype=np.float32)/255.0

    sh_gain = 0.22 + 0.35*strength   # подъём теней
    hl_cut  = 0.15 + 0.25*strength   # срез хайлайтов

    for c in range(3):
        chan = arr[...,c]
        chan = chan + sh_mask * sh_gain * (1.0 - chan)  # приподнять тени
        chan = chan - hl_mask * hl_cut * chan           # приглушить хайлайты
        arr[...,c] = np.clip(chan, 0.0, 1.0)

    base = Image.fromarray((arr*255).astype(np.uint8))

    # локальный контраст (высокочастотная составляющая)
    blurred = base.filter(ImageFilter.GaussianBlur(radius=1.8 + 3.5*strength))
    hp = ImageChops.subtract(base, blurred)
    hp = hp.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(120+120*strength), threshold=3))
    mc_amount = 0.20 + 0.25*strength
    base = Image.blend(base, hp, mc_amount)

    # лёгкая общая резкость
    base = base.filter(ImageFilter.UnsharpMask(radius=1.2, percent=100+int(100*strength), threshold=2))
    # лёгкая насыщенность
    sat = 1.04 + 0.20*strength
    base = ImageEnhance.Color(base).enhance(sat)

    fd, out_path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    base.save(out_path, "JPEG", quality=95, optimize=True)
    return out_path
# === /добавлено ===

# ---------- PIPELINES ----------
# 1) Nature Enhance — Clarity + HDR
# === добавлено ===
async def run_nature_enhance_clarity_hdr(file_id: str, strength: float) -> str:
    """
    TG URL -> CLARITY (Replicate) -> локальный HDR тонмаппинг -> возвращаем локальный путь к файлу
    """
    public_url = await telegram_file_to_public_url(file_id)

    prompt_text = (
        "masterpiece, best quality, highres,\n"
        "<lora:more_details:0.5>\n"
        "<lora:SDXLrender_v2.0:1>"
    )
    negative = "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"

    inputs = {
        "image": public_url,
        "prompt": prompt_text,
        "negative_prompt": negative,
        "scale_factor": 2,
        "dynamic": 6,
        "creativity": 0.25,
        "resemblance": 0.65,
        "tiling_width": 112,
        "tiling_height": 144,
        "sd_model": "juggernaut_reborn.safetensors [338b85bc4f]",
        "scheduler": "DPM++ 3M SDE Karras",
        "num_inference_steps": 22,
        "seed": 1337,
        "downscaling": False,
        "sharpen": 0,
        "handfix": "disabled",
        "output_format": "png",
    }

    out = replicate.run(MODEL_CLARITY, input=inputs)
    cl_url = pick_first_url(out)

    # HDR поверх Clarity-результата
    cl_path = download_to_temp(cl_url)
    try:
        hdr_path = hdr_enhance_path(cl_path, strength=strength)
        return hdr_path  # локальный файл
    finally:
        try: os.remove(cl_path)
        except: pass
# === /добавлено ===

# 2) Nature Enhance 2.0 — только HDR (без Clarity)
# === добавлено ===
async def run_nature_enhance_hdr_only(file_id: str, strength: float) -> str:
    """
    TG URL -> скачиваем исходник -> локальный HDR тонмаппинг -> возвращаем локальный путь к файлу
    """
    public_url = await telegram_file_to_public_url(file_id)
    src_path = download_to_temp(public_url)
    try:
        hdr_path = hdr_enhance_path(src_path, strength=strength)
        return hdr_path
    finally:
        try: os.remove(src_path)
        except: pass
# === /добавлено ===

# ---------- UI ----------
# === изменено: новое меню с двумя режимами и выбором силы ===
KB_MAIN = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance (Clarity + HDR)")],
        [KeyboardButton("🌿 Nature Enhance 2.0 (HDR)")],
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
# === /изменено ===

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer(
        "Nature Inspire готово 🌿\n"
        "Выбери режим и силу эффекта:\n"
        "• Nature Enhance — Clarity + натуральный HDR\n"
        "• Nature Enhance 2.0 — только натуральный HDR\n",
        reply_markup=KB_MAIN
    )

@dp.message_handler(lambda m: m.text in [
    "🌿 Nature Enhance (Clarity + HDR)",
    "🌿 Nature Enhance 2.0 (HDR)"
])
async def on_choose_mode(m: types.Message):
    uid = m.from_user.id
    if "Clarity" in m.text:
        WAIT[uid] = {"effect": "nature_menu"}      # выберем силу, затем ждём фото
    else:
        WAIT[uid] = {"effect": "nature2_menu"}     # выберем силу, затем ждём фото
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

    # силу переведём в 0..1
    strength = 0.6
    if m.text == "Низкая":  strength = 0.35
    if m.text == "Средняя": strength = 0.6
    if m.text == "Высокая": strength = 0.85

    if st["effect"] == "nature_menu":
        WAIT[uid] = {"effect": "nature_clarity_hdr", "strength": strength}
        await m.answer("Пришли фото — сделаю Nature Enhance (Clarity + HDR) 🌿", reply_markup=KB_MAIN)
    elif st["effect"] == "nature2_menu":
        WAIT[uid] = {"effect": "nature_hdr", "strength": strength}
        await m.answer("Пришли фото — сделаю Nature Enhance 2.0 (HDR) 🌿", reply_markup=KB_MAIN)

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st or st.get("effect") not in ["nature_clarity_hdr", "nature_hdr"]:
        await m.reply("Сначала выбери режим и силу эффекта ⬇️", reply_markup=KB_MAIN)
        return

    await m.reply("⏳ Обрабатываю...")
    try:
        effect = st["effect"]
        strength = float(st.get("strength", 0.6))

        if effect == "nature_clarity_hdr":
            out_path = await run_nature_enhance_clarity_hdr(m.photo[-1].file_id, strength=strength)
        else:
            out_path = await run_nature_enhance_hdr_only(m.photo[-1].file_id, strength=strength)

        # отправляем локальный файл как фото
        safe_path = ensure_photo_size_under_telegram_limit(out_path)
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
