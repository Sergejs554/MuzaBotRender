# bot.py — Nature Inspire (Replicate)
# ENV:
#   TELEGRAM_API_TOKEN=xxxx:yyyy
#   REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxx

import os
import logging
import traceback
import tempfile
import aiohttp
import replicate
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor

logging.basicConfig(level=logging.INFO)

# ---------- TOKENS ----------
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
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

def tg_file_url(file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

def pick_url(output):
    # Универсально достаём URL из ответа Replicate
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

# ---------- HELPERS ----------
async def download_telegram_file_to_temp(file_id: str) -> str:
    """Скачиваем фото из Telegram во временный файл и возвращаем путь."""
    tg_file = await bot.get_file(file_id)
    url = tg_file_url(tg_file.file_path)
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return tmp_path

# ===================== PIPELINES =====================

def run_nature_enhance_from_path(local_path: str) -> str:
    """🌿 Magic Image Refiner -> ESRGAN x2 (вход — локальный файл)."""
    # шаг 1: рефайн (передаём как file-object — SDK сам зальёт)
    with open(local_path, "rb") as f:
        ref_out = replicate.run(
            MODEL_REFINER,
            input={
                "image": f,
                # мягкий автопромпт; при желании поднимай/меняй
                "prompt": "natural color balance, clean details, no artifacts, no extra objects"
            }
        )
    ref_url = pick_url(ref_out)

    # шаг 2: апскейл x2
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": ref_url, "scale": 2})
    return pick_url(esr_out)

def run_epic_landscape_flux(prompt_text: str) -> str:
    """🌄 Текст -> картинка (Flux)."""
    if not prompt_text or not prompt_text.strip():
        prompt_text = ("epic panoramic landscape, dramatic sky, volumetric light, "
                       "ultra-detailed mountains, lush forests, cinematic composition, "
                       "award-winning nature photography")
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    return pick_url(flux_out)

def run_ultra_hdr(prompt_text: str) -> str:
    """🏞 «HDR усиление» через Flux по тексту + апскейл x2.
    (Flux не умеет image2image, поэтому всегда нужна подсказка.)"""
    if not prompt_text or not prompt_text.strip():
        prompt_text = "Ultra HDR, realistic nature photo of the same scene, high dynamic range, crisp details"
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    flux_url = pick_url(flux_out)
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": flux_url, "scale": 2})
    return pick_url(esr_out)

def run_clean_restore_from_path(local_path: str) -> str:
    """📸 SwinIR очистка -> ESRGAN x2 (вход — локальный файл)."""
    with open(local_path, "rb") as f:
        swin_out = replicate.run(MODEL_SWINIR, input={"image": f, "jpeg": "40", "noise": "15"})
    swin_url = pick_url(swin_out)
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": swin_url, "scale": 2})
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
        "Выбери режим ниже. Для Flux можно прислать только текст (описание).",
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
        await m.answer("Пришли описание пейзажа текстом (или фото с подписью).")
    elif "Ultra HDR" in m.text:
        WAIT[uid] = {"effect": "hdr"}
        await m.answer("Пришли фото с короткой подписью сцены — усилю в HDR-стиле.")
    elif "Clean Restore" in m.text:
        WAIT[uid] = {"effect": "clean"}
        await m.answer("Пришли фото — аккуратно почищу и детализирую.")

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    state = WAIT.get(uid)
    if not state:
        await m.reply("Выбери режим на клавиатуре ниже и затем пришли фото.", reply_markup=KB)
        return

    effect = state.get("effect", "?")
    caption = (m.caption or "").strip()

    await m.reply("⏳ Обрабатываю...")
    tmp_path = None
    try:
        # Скачиваем телеграм-файл во временный путь
        tmp_path = await download_telegram_file_to_temp(m.photo[-1].file_id)

        if effect == "nature":
            out_url = run_nature_enhance_from_path(tmp_path)
        elif effect == "hdr":
            out_url = run_ultra_hdr(caption)
        elif effect == "clean":
            out_url = run_clean_restore_from_path(tmp_path)
        elif effect == "flux":
            # Если человек отправил фото в режиме Flux — используем подпись как промпт
            out_url = run_epic_landscape_flux(prompt_text=caption)
        else:
            raise RuntimeError(f"Unknown effect: {effect}")

        await m.reply_photo(out_url)

    except Exception:
        tb = traceback.format_exc(limit=30)
        # Шлём стек в чат, как просил
        await m.reply(f"🔥 Ошибка {effect}:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        WAIT.pop(uid, None)

# Текстовый хендлер для Flux (когда человек шлёт только описания)
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
        await m.reply_photo(out_url)
    except Exception:
        tb = traceback.format_exc(limit=30)
        await m.reply(f"🔥 Ошибка flux:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling...")
    executor.start_polling(dp, skip_updates=True)
