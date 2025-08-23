# bot.py — Nature Inspire (Replicate) — ULTRA HDR Enhance

import os
import logging
import replicate
import traceback
import urllib.request
import tempfile
from PIL import Image

from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InputFile
from aiogram.utils import executor

logging.basicConfig(level=logging.INFO)

# ---------- TOKENS ----------
API_TOKEN  = os.getenv("TELEGRAM_API_TOKEN")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("TELEGRAM_API_TOKEN")
if not REPL_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ---------- MODELS ----------
MODEL_FLUX    = "black-forest-labs/flux-1.1-pro"
MODEL_REFINER = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_ESRGAN  = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
MODEL_SWINIR  = "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"

# ---------- STATE ----------
WAIT = {}  # user_id -> {'effect': ...}

# ---------- HELPERS ----------
def tg_public_url(file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

async def telegram_file_to_public_url(file_id: str) -> str:
    tg_file = await bot.get_file(file_id)
    return tg_public_url(tg_file.file_path)

def pick_url(output) -> str:
    try:
        if isinstance(output, str):
            return output
        if isinstance(output, (list, tuple)) and output:
            o0 = output[0]
            return getattr(o0, "url", str(o0))
        return getattr(output, "url", str(output))
    except Exception:
        return str(output)

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

async def send_image_by_url(m: types.Message, url: str):
    """Скачиваем и шлём как файл, чтобы TG не ругался на url."""
    path = None
    try:
        path = download_to_temp(url)
        # fail-safe resize если >10MB
        if os.path.getsize(path) > 9_500_000:
            img = Image.open(path)
            img.thumbnail((1600, 1600), Image.LANCZOS)
            img.save(path, "JPEG", quality=95)
        await m.reply_photo(InputFile(path))
    finally:
        if path and os.path.exists(path):
            os.remove(path)

async def download_and_resize_input(file_id: str, max_side: int = 1280) -> str:
    """Скачиваем фото из TG и уменьшаем до безопасного размера (по длинной стороне)."""
    tg_file = await bot.get_file(file_id)
    url = tg_public_url(tg_file.file_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(url, tmp_path)
    img = Image.open(tmp_path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(tmp_path, "JPEG", quality=95)
    return tmp_path

# ===================== PIPELINES =====================

def run_nature_enhance_pipeline(local_path: str) -> str:
    """
    🌿 Nature Enhance = Refiner(clean) -> Refiner(HDR) -> ESRGAN x4 -> safe resize
    """
    # 1) первый Refiner (чистка/баланс)
    ref1_out = replicate.run(
        MODEL_REFINER,
        input={
            "image": open(local_path, "rb"),
            "prompt": "natural clean enhancement, remove artifacts, balance exposure, keep details"
        }
    )
    ref1_url = pick_url(ref1_out)

    # 2) второй Refiner (HDR динамика и сочность)
    ref2_out = replicate.run(
        MODEL_REFINER,
        input={
            "image": ref1_url,
            "prompt": "Ultra HDR vivid colors, deep dynamic range, high detail preservation, cinematic look"
        }
    )
    ref2_url = pick_url(ref2_out)

    # 3) upscale ESRGAN x4
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": ref2_url, "scale": 4})
    esr_url = pick_url(esr_out)

    return esr_url

# ===================== UI / HANDLERS =====================

KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance")],
        [KeyboardButton("🌄 Epic Landscape Flux")],
        [KeyboardButton("🏞 Ultra HDR")],
        [KeyboardButton("📸 Clean Restore")],
    ],
    resize_keyboard=True
)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer("Привет ✨ Природу сделаю в стиле *Ultra HDR Enhance*.\nВыбирай режим.", reply_markup=KB)

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance", "🌄 Epic Landscape Flux", "🏞 Ultra HDR", "📸 Clean Restore"])
async def on_choose(m: types.Message):
    uid = m.from_user.id
    if "Nature Enhance" in m.text:
        WAIT[uid] = {"effect": "nature"}
        await m.answer("Пришли фото 🌿 — сделаю Ultra HDR Enhance ✨")
    else:
        WAIT[uid] = {"effect": "skip"}
        await m.answer("Эффект в доработке. Ждём 🌙")

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st:
        await m.reply("Сначала выбери режим на клавиатуре ⬇️", reply_markup=KB)
        return
    effect = st["effect"]

    await m.reply("⏳ Обрабатываю фото...")

    try:
        if effect == "nature":
            tmp_path = await download_and_resize_input(m.photo[-1].file_id, 1280)
            out_url = run_nature_enhance_pipeline(tmp_path)
            await send_image_by_url(m, out_url)
            os.remove(tmp_path)
        else:
            await m.reply("Эффект пока не активирован 🔧")
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка {effect}:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling...")
    executor.start_polling(dp, skip_updates=True)
