# bot.py — Nature Inspire (Replicate) — FULL: double refiner + ESRGAN x4 + safe IO

import os
import logging
import asyncio
import traceback
import tempfile
import urllib.request

import replicate
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InputFile
from aiogram.utils import executor
from PIL import Image

logging.basicConfig(level=logging.INFO)

# ---------- TOKENS ----------
API_TOKEN  = os.getenv("TELEGRAM_API_TOKEN")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("TELEGRAM_API_TOKEN")
if not REPL_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN  # для SDK

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ---------- MODELS ----------
MODEL_FLUX    = "black-forest-labs/flux-1.1-pro"
MODEL_REFINER = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_ESRGAN  = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"

# ---------- STATE ----------
WAIT = {}  # user_id -> {'effect': 'nature'|'flux'|'hdr'|'clean'}

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
            if hasattr(o0, "url"):
                return o0.url if isinstance(o0.url, str) else str(o0.url)
            return str(o0)
        if hasattr(output, "url"):
            return output.url if isinstance(output.url, str) else str(output.url)
        return str(output)
    except Exception:
        return str(output)

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def compress_under_telegram_limit(path_in: str, max_bytes: int = 10 * 1024 * 1024) -> str:
    """
    Если файл >10MB, пережимаем JPEG (и при необходимости уменьшаем сторону),
    чтобы уложиться в лимит Telegram для фото.
    """
    if os.path.getsize(path_in) <= max_bytes:
        return path_in

    img = Image.open(path_in).convert("RGB")
    w, h = img.size

    # Пошагово снижаем качество и, если надо, размер
    quality = 92
    step_q = 8
    max_side = max(w, h)

    while True:
        fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        # если всё ещё слишком большой — уменьшаем габарит
        if max_side > 2048:
            scale = 2048 / float(max_side)
            img2 = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            img2 = img

        img2.save(out_path, "JPEG", quality=quality, optimize=True, progressive=True)
        size = os.path.getsize(out_path)

        if size <= max_bytes or quality <= 40:
            # готово, удалим исходник, вернём путь на сжатый
            try:
                if path_in != out_path and os.path.exists(path_in):
                    os.remove(path_in)
            except Exception:
                pass
            return out_path

        # не уложились — усилить компрессию
        try:
            os.remove(out_path)
        except Exception:
            pass
        quality -= step_q
        if quality < 40 and max_side > 1600:
            max_side = 1600  # дополнительное уменьшение
        elif quality < 40 and max_side > 1280:
            max_side = 1280

async def send_image_by_url(m: types.Message, url: str):
    """
    Надёжная отправка: скачиваем по URL → при необходимости сжимаем → отправляем как файл.
    """
    path = None
    try:
        path = download_to_temp(url)
        path = compress_under_telegram_limit(path)  # уложим в 10MB
        await m.reply_photo(InputFile(path))
    finally:
        if path and os.path.exists(path):
            try: os.remove(path)
            except: pass

async def download_and_resize_input(file_id: str, max_side: int = 1280) -> str:
    """
    Скачиваем входной кадр из Telegram и приводим к безопасному размеру (по длинной стороне)
    чтобы не словить GPU‑лимит на стороне Replicate.
    Возвращает путь к локальному JPG.
    """
    tg_url = await telegram_file_to_public_url(file_id)
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(tg_url, tmp)

    img = Image.open(tmp).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(tmp, "JPEG", quality=95)
    return tmp

# ===================== PIPELINES =====================

def run_double_refiner_then_esrgan4(path_or_url_for_ref1) -> str:
    """
    Pass1 Refiner (clean/balance) -> Pass2 Refiner (Ultra HDR) -> ESRGAN x4
    Первый проход принимает локальный файл (безопасно по размеру),
    второй берёт URL из первого прохода.
    """
    # --- Refiner Pass 1: мягкая чистка/баланс ---
    with (open(path_or_url_for_ref1, "rb") if os.path.exists(path_or_url_for_ref1) else None) as f1:
        ref1_in = {
            "image": f1 if f1 else path_or_url_for_ref1,
            "prompt": "natural color balance, remove artifacts, clearer details, preserve realism"
        }
        ref1_out = replicate.run(MODEL_REFINER, input=ref1_in)
    ref1_url = pick_url(ref1_out)

    # --- Refiner Pass 2: усиление HDR/цвет/объём ---
    ref2_in = {
        "image": ref1_url,
        "prompt": "Ultra HDR, rich dynamic range, deep shadows, highlight recovery, vibrant yet realistic colors, cinematic depth, crisp micro-contrast"
    }
    ref2_out = replicate.run(MODEL_REFINER, input=ref2_in)
    ref2_url = pick_url(ref2_out)

    # --- ESRGAN x4: детализация/резкость ---
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": ref2_url, "scale": 4})
    return pick_url(esr_out)

def run_nature_enhance_pipeline(file_id: str) -> str:
    """
    Полный Nature Enhance:
    - входное фото → resize до 1280px (локально)
    - Refiner Pass1 → Refiner Pass2 → ESRGAN x4
    - возвращаем финальный URL
    """
    tmp_path = asyncio.get_event_loop().run_until_complete(download_and_resize_input(file_id, 1280))
    try:
        final_url = run_double_refiner_then_esrgan4(tmp_path)
        return final_url
    finally:
        try: os.remove(tmp_path)
        except: pass

def run_epic_landscape_flux(prompt_text: str) -> str:
    if not prompt_text or not prompt_text.strip():
        prompt_text = (
            "epic panoramic landscape, dramatic sky, volumetric light, ultra-detailed mountains, "
            "lush forests, cinematic composition, award-winning nature photography"
        )
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    return pick_url(flux_out)

def run_ultra_hdr_text(_public_url_ignored: str, hint_caption: str = "") -> str:
    """
    Текстовый HDR (если оставляем кнопку) — на базе FLUX по описанию, затем ESRGAN x4.
    """
    prompt_text = hint_caption.strip() if hint_caption else (
        "Ultra HDR nature photo of the same scene, rich dynamic range, crisp details, "
        "deep shadows, highlight recovery, realistic colors, professional photography"
    )
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    flux_url = pick_url(flux_out)

    esr_out = replicate.run(MODEL_ESRGAN, input={"image": flux_url, "scale": 4})
    return pick_url(esr_out)

# ===================== UI / HANDLERS =====================

KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance")],
        [KeyboardButton("🌄 Epic Landscape Flux")],
        [KeyboardButton("🏞 Ultra HDR")],
    ],
    resize_keyboard=True
)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer(
        "Природа на максимум. Выбери режим и пришли фото (для Flux — можно просто текст).",
        reply_markup=KB
    )

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance", "🌄 Epic Landscape Flux", "🏞 Ultra HDR"])
async def on_choose(m: types.Message):
    uid = m.from_user.id
    if "Nature Enhance" in m.text:
        WAIT[uid] = {"effect": "nature"}
        await m.answer("Ок! Пришли фото. ⛰️🌿")
    elif "Epic Landscape Flux" in m.text:
        WAIT[uid] = {"effect": "flux"}
        await m.answer("Пришли подпись-описание пейзажа (или просто текст без фото) — сгенерю кадр.")
    elif "Ultra HDR" in m.text:
        WAIT[uid] = {"effect": "hdr"}
        await m.answer("Пришли фото. Можно приложить подпись — опишешь сцену; усилю в стиле HDR.")

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st:
        await m.reply("Выбери режим на клавиатуре ниже и затем пришли фото.", reply_markup=KB)
        return

    effect = st["effect"]
    caption = (m.caption or "").strip()
    await m.reply("⏳ Обрабатываю...")

    try:
        if effect == "nature":
            out_url = run_nature_enhance_pipeline(m.photo[-1].file_id)
            await send_image_by_url(m, out_url)
        elif effect == "flux":
            out_url = run_epic_landscape_flux(prompt_text=caption)
            await send_image_by_url(m, out_url)
        elif effect == "hdr":
            out_url = run_ultra_hdr_text("", hint_caption=caption)
            await send_image_by_url(m, out_url)
        else:
            raise RuntimeError("Unknown effect")
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка {effect}:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

@dp.message_handler(content_types=["text"])
async def on_text(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st or st.get("effect") != "flux":
        return
    prompt = m.text.strip()
    await m.reply("⏳ Генерирую пейзаж по описанию…")
    try:
        out_url = run_epic_landscape_flux(prompt_text=prompt)
        await send_image_by_url(m, out_url)
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка flux:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling...")
    executor.start_polling(dp, skip_updates=True)
