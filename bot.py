# bot.py — Nature Inspire (Replicate)

import os
import logging
import traceback
import tempfile
import urllib.request

import replicate
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor

logging.basicConfig(level=logging.INFO)

# ===== TOKENS =====
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("TELEGRAM_API_TOKEN")
if not REPL_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN  # для SDK

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ===== MODELS =====
MODEL_FLUX    = "black-forest-labs/flux-1.1-pro"
MODEL_REFINER = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_ESRGAN  = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
MODEL_SWINIR  = "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"

# ===== STATE =====
WAIT = {}  # user_id -> {'effect': ...}

def tg_file_url(file_path: str) -> str:
    """Собрать прямой URL на файл Telegram (для скачивания нашим кодом)."""
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

def pick_url(output):
    """Аккуратно достаём URL из ответа Replicate (строка / список / Blob с .url)."""
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

def telegram_file_to_replicate_url(file_id: str) -> str:
    """
    Скачиваем фото из Telegram во временный файл и загружаем на Replicate Delivery,
    возвращаем стабильный https-URL (подходит абсолютно для всех моделей).
    """
    tmp_path = None
    try:
        tg_file = bot.loop.run_until_complete(bot.get_file(file_id))
        public_src = tg_file_url(tg_file.file_path)

        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        urllib.request.urlretrieve(public_src, tmp_path)

        # Важно: используем официальный аплоад SDK
        uploaded = replicate.files.upload(tmp_path)  # -> https://replicate.delivery/...
        return uploaded
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ===================== PIPELINES =====================

def run_nature_enhance(replicate_url: str) -> str:
    """🌿 Refiner → ESRGAN x2."""
    ref_inputs = {
        "image": replicate_url,
        "prompt": "natural color balance, clean details, no artifacts, no extra objects"
        # если появится параметр силы у конкретной версии — добавим его здесь
    }
    ref_out = replicate.run(MODEL_REFINER, input=ref_inputs)
    ref_url = pick_url(ref_out)

    esr_out = replicate.run(MODEL_ESRGAN, input={"image": ref_url, "scale": 2})
    return pick_url(esr_out)

def run_epic_landscape_flux(prompt_text: str) -> str:
    """🌄 Генерация по тексту (без входного фото)."""
    if not prompt_text or not prompt_text.strip():
        prompt_text = (
            "epic panoramic landscape, dramatic sky, volumetric light, ultra-detailed mountains, "
            "lush forests, cinematic composition, award-winning nature photography"
        )
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    return pick_url(flux_out)

def run_ultra_hdr(hint_caption: str = "") -> str:
    """🏞 Flux с HDR-подсказкой → ESRGAN x2."""
    prompt_text = hint_caption.strip() if hint_caption else "Ultra HDR, realistic nature photo, same scene look, rich dynamic range"
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    flux_url = pick_url(flux_out)

    esr_out = replicate.run(MODEL_ESRGAN, input={"image": flux_url, "scale": 2})
    return pick_url(esr_out)

def run_clean_restore(replicate_url: str) -> str:
    """📸 SwinIR (мягкая чистка) → ESRGAN x2."""
    swin_out = replicate.run(
        MODEL_SWINIR,
        input={"image": replicate_url, "jpeg": "40", "noise": "15"}
    )
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
        "Выбери режим ниже. Для Flux можно прислать только текст (без фото).",
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
        await m.answer("Пришли *текст-описание* пейзажа (или просто отправь текстом) — сгенерю кадр.", parse_mode="Markdown")
    elif "Ultra HDR" in m.text:
        WAIT[uid] = {"effect": "hdr"}
        await m.answer("Пришли фото (можно с короткой подписью, например: «яркое HDR небо, сочная зелень»).")
    elif "Clean Restore" in m.text:
        WAIT[uid] = {"effect": "clean"}
        await m.answer("Пришли фото — аккуратно почищу шум/мыло и детализирую.")

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
        # 1) конвертируем телеграм-файл -> стабильный replicate.delivery URL
        rep_url = telegram_file_to_replicate_url(m.photo[-1].file_id)

        # 2) запускаем нужный пайплайн
        if effect == "nature":
            out_url = run_nature_enhance(rep_url)
        elif effect == "hdr":
            out_url = run_ultra_hdr(hint_caption=caption)
        elif effect == "clean":
            out_url = run_clean_restore(rep_url)
        elif effect == "flux":
            # Если человек всё-таки прислал фото, берём подпись как промпт.
            out_url = run_epic_landscape_flux(prompt_text=caption)
        else:
            raise RuntimeError("Unknown effect")

        await m.reply_photo(out_url)

    except Exception:
        tb = traceback.format_exc(limit=40)
        await m.reply(f"🔥 Ошибка {effect}:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

@dp.message_handler(content_types=["text"])
async def on_text(m: types.Message):
    uid = m.from_user.id
    state = WAIT.get(uid)
    if not state or state.get("effect") != "flux":
        return  # реагируем только на Flux

    prompt = (m.text or "").strip()
    await m.reply("⏳ Генерирую пейзаж по описанию...")
    try:
        out_url = run_epic_landscape_flux(prompt_text=prompt)
        await m.reply_photo(out_url)
    except Exception:
        tb = traceback.format_exc(limit=40)
        await m.reply(f"🔥 Ошибка flux:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling...")
    executor.start_polling(dp, skip_updates=True)
