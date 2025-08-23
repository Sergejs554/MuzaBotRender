# bot.py — Nature Inspire (Replicate) — async resize + double refiner + ESRGAN x4 + safe send

import os
import logging
import replicate
import asyncio
import traceback
import aiohttp
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
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN  # для SDK

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ---------- MODELS ----------
MODEL_FLUX    = "black-forest-labs/flux-1.1-pro"
MODEL_REFINER = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_ESRGAN  = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
MODEL_SWINIR  = "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"

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
            url_attr = getattr(o0, "url", None)
            return (url_attr() if callable(url_attr) else url_attr) or str(o0)
        url_attr = getattr(output, "url", None)
        return (url_attr() if callable(url_attr) else url_attr) or str(output)
    except Exception:
        return str(output)

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

async def send_image_by_url(m: types.Message, url: str):
    """Скачиваем итог в temp и шлём как файл — надёжно для Telegram."""
    path = None
    try:
        path = download_to_temp(url)
        # Telegram лимит фото = 10MB. Если вдруг больше — ужмём.
        if os.path.getsize(path) > 10 * 1024 * 1024:
            img = Image.open(path).convert("RGB")
            img.save(path, "JPEG", quality=88, optimize=True)
        await m.reply_photo(InputFile(path))
    finally:
        if path and os.path.exists(path):
            os.remove(path)

# ---------- ASYNC RESIZE ----------
async def download_and_resize_input(file_id: str, max_side: int = 1280) -> str:
    """
    Скачиваем фото из Telegram и безопасно уменьшаем по длинной стороне до max_side.
    Возвращаем путь к временному JPG.
    """
    tg_file = await bot.get_file(file_id)
    url = tg_public_url(tg_file.file_path)

    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            r.raise_for_status()
            data = await r.read()

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(data)

    img = Image.open(tmp_path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(tmp_path, "JPEG", quality=95, optimize=True)

    return tmp_path

# ===================== PIPELINES =====================

async def run_nature_enhance_pipeline(file_id: str) -> str:
    """
    🌿 Nature Enhance (ТОП-версия):
      1) async resize входа (GPU‑safe)
      2) Refiner #1 — чистка/баланс/детали
      3) Refiner #2 — HDR‑динамика/глубина/сочность (агрессивнее)
      4) ESRGAN x4 — детализация (fallback на x2 при OOM)
    """
    # 1) safe resize
    tmp_path = await download_and_resize_input(file_id, max_side=1280)

    # 2) Refiner pass #1 (мягко, натурально)
    ref1_out = replicate.run(
        MODEL_REFINER,
        input={
            "image": open(tmp_path, "rb"),
            "prompt": "natural color balance, realistic contrast, preserve textures, remove artifacts"
        }
    )
    ref1_url = pick_url(ref1_out)

    # 3) Refiner pass #2 (HDR mood)
    ref2_out = replicate.run(
        MODEL_REFINER,
        input={
            "image": ref1_url,
            "prompt": "ULTRA HDR look, deep rich colors, wide dynamic range, crisp micro-contrast, vivid yet realistic"
        }
    )
    ref2_url = pick_url(ref2_out)

    # 4) ESRGAN upscale x4 с фолбэком на x2 при нехватке VRAM
    try:
        esr_out = replicate.run(MODEL_ESRGAN, input={"image": ref2_url, "scale": 4})
    except Exception:
        esr_out = replicate.run(MODEL_ESRGAN, input={"image": ref2_url, "scale": 2})
    final_url = pick_url(esr_out)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return final_url

def run_epic_landscape_flux(prompt_text: str) -> str:
    if not prompt_text or not prompt_text.strip():
        prompt_text = (
            "epic panoramic landscape, dramatic sky, volumetric light, ultra-detailed mountains, "
            "lush forests, cinematic composition, award-winning nature photography"
        )
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    return pick_url(flux_out)

def run_ultra_hdr(_public_url_ignored: str, hint_caption: str = "") -> str:
    prompt_text = hint_caption.strip() if hint_caption else (
        "Ultra HDR nature photo of the same scene, rich dynamic range, crisp details, "
        "deep shadows, highlight recovery, realistic colors, professional photography"
    )
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    flux_url = pick_url(flux_out)
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": flux_url, "scale": 4})
    return pick_url(esr_out)

def run_clean_restore(public_url: str) -> str:
    swin_out = replicate.run(MODEL_SWINIR, input={"image": public_url, "jpeg": "40", "noise": "15"})
    swin_url = pick_url(swin_out)
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": swin_url, "scale": 4})
    return pick_url(esr_out)

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
    await m.answer(
        "Привет ✨ Природные кадры улучшим на максимум.\n"
        "Выбери режим ниже, затем пришли фото (для Flux-генерации можно прислать только текст в подписи).",
        reply_markup=KB
    )

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance", "🌄 Epic Landscape Flux", "🏞 Ultra HDR", "📸 Clean Restore"])
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
        await m.answer("Пришли фото. Можно приложить подпись — опишешь сцену, усилю её в стиле HDR.")
    elif "Clean Restore" in m.text:
        WAIT[uid] = {"effect": "clean"}
        await m.answer("Пришли фото. Уберу шум/мыло и аккуратно детализирую.")

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    state = WAIT.get(uid)
    if not state:
        await m.reply("Выбери режим на клавиатуре ниже и затем пришли фото.", reply_markup=KB)
        return

    effect = state.get("effect")
    caption = (m.caption or "").strip()

    await m.reply("⏳ Обрабатываю...")
    try:
        # Nature Enhance — async пайплайн (ВАЖНО: await!)
        if effect == "nature":
            out_url = await run_nature_enhance_pipeline(m.photo[-1].file_id)
        elif effect == "flux":
            out_url = run_epic_landscape_flux(prompt_text=caption)
        elif effect == "hdr":
            # Фото сейчас не используем; работаем через подсказку
            out_url = run_ultra_hdr("", hint_caption=caption)
        elif effect == "clean":
            public_url = await telegram_file_to_public_url(m.photo[-1].file_id)
            out_url = run_clean_restore(public_url)
        else:
            raise RuntimeError("Unknown effect")

        await send_image_by_url(m, out_url)

    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка {effect}:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

# Текстовая генерация для Flux без фото
@dp.message_handler(content_types=["text"])
async def on_text(m: types.Message):
    uid = m.from_user.id
    state = WAIT.get(uid)
    if not state or state.get("effect") != "flux":
        return
    prompt = m.text.strip()
    await m.reply("⏳ Генерирую пейзаж по описанию...")
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
