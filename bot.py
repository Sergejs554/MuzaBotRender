# bot.py — Nature Inspire (Replicate)
# Требуются переменные окружения:
#   TELEGRAM_API_TOKEN=xxxx:yyyy
#   REPLICATE_API_TOKEN=r8_************

import os
import logging
import replicate
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor

logging.basicConfig(level=logging.INFO)

# ----- TOKENS -----
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("TELEGRAM_API_TOKEN")
if not REPL_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN  # для SDK

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ----- MODELS -----
MODEL_FLUX        = "black-forest-labs/flux-1.1-pro"  # текст→картинка
MODEL_REFINER     = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_ESRGAN      = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
MODEL_SWINIR      = "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"

# ----- STATE -----
WAIT = {}  # user_id -> {'effect': ...}

def tg_file_url(file_path: str) -> str:
    """Публичный URL фото из Telegram (Replicate принимает только URL)."""
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

def pick_url(output):
    """Надёжно достаем URL из разных форматов ответа Replicate."""
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

# ===================== PIPELINES =====================

def run_nature_enhance(public_url: str) -> str:
    """
    🌿 Nature Enhance = Magic Image Refiner (улучшение) -> ESRGAN x2 (апскейл).
    """
    # шаг 1: рефайнер
    ref_inputs = {
        "image": public_url,
        # можешь тюнить промпт; по умолчанию нейтрально улучшает
        "prompt": "natural color balance, clean details, no artifacts, no extra objects"
    }
    ref_out = replicate.run(MODEL_REFINER, input=ref_inputs)
    ref_url = pick_url(ref_out)

    # шаг 2: апскейл/детализация
    esr_inputs = {"image": ref_url, "scale": 2}
    esr_out = replicate.run(MODEL_ESRGAN, input=esr_inputs)
    return pick_url(esr_out)

def run_epic_landscape_flux(prompt_text: str) -> str:
    """
    🌄 Epic Landscape Flux = чистая генерация по тексту (без входного фото).
    """
    if not prompt_text or not prompt_text.strip():
        prompt_text = (
            "epic panoramic landscape, dramatic sky, volumetric light, ultra-detailed mountains, "
            "lush forests, cinematic composition, award-winning nature photography"
        )
    flux_inputs = {
        "prompt": prompt_text,
        "prompt_upsampling": True
    }
    flux_out = replicate.run(MODEL_FLUX, input=flux_inputs)
    return pick_url(flux_out)

def run_ultra_hdr(public_url: str, hint_caption: str = "") -> str:
    """
    🏞 Ultra HDR = Flux 'image-inspired' через подсказку (caption как описание сцены) → ESRGAN x2.
    (Flux здесь используется как мощный 'перерисовщик настроения' по тексту-описанию.)
    """
    # На FLUX 1.1 pro нет прямого image2image, поэтому используем caption как направляющий промпт.
    # Если caption пуст — подставим HDR-шаблон.
    prompt_text = hint_caption.strip() if hint_caption else (
        "HDR nature photo of the same scene, rich dynamic range, crisp details, deep shadows, "
        "highlight recovery, realistic colors, professional nature photography"
    )
    # Генерация новая (не детермин. ремастер), зато качество + потом апскейл:
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    flux_url = pick_url(flux_out)

    esr_out = replicate.run(MODEL_ESRGAN, input={"image": flux_url, "scale": 2})
    return pick_url(esr_out)

def run_clean_restore(public_url: str) -> str:
    """
    📸 Clean Restore = SwinIR (убрать шум/жесть) → ESRGAN x2 (детализация).
    """
    # SwinIR ждёт параметры 'jpeg' и 'noise' как строки; оставим мягкие значения.
    swin_inputs = {
        "image": public_url,
        "jpeg": "40",   # степень jpeg-деградации (модели так привычнее)
        "noise": "15"   # уровень шума
    }
    swin_out = replicate.run(MODEL_SWINIR, input=swin_inputs)
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
        await m.answer("Пришли фото (по желанию) с подписью-описанием сцены — возьму подпись как промпт. Если подпись пустая, сгенерю эпик-ландшафт по умолчанию.")
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

    # Получаем Telegram-file URL
    await m.reply("⏳ Обрабатываю...")
    try:
        tg_file = await bot.get_file(m.photo[-1].file_id)
        public_url = tg_file_url(tg_file.file_path)

        if effect == "nature":
            out_url = run_nature_enhance(public_url)
        elif effect == "flux":
            # Для Flux берём caption как промпт. Фото не используем напрямую.
            out_url = run_epic_landscape_flux(prompt_text=caption)
        elif effect == "hdr":
            out_url = run_ultra_hdr(public_url, hint_caption=caption)
        elif effect == "clean":
            out_url = run_clean_restore(public_url)
        else:
            raise RuntimeError("Unknown effect")

        await m.reply_photo(out_url)

    except Exception:
        # Без техподробностей
        await m.reply("Не удалось обработать фото. Попробуй другую фотографию или другую подпись.")
    finally:
        WAIT.pop(uid, None)

# Также разрешим чисто текст для Flux (пользователь может прислать prompt без фото)
@dp.message_handler(content_types=["text"])
async def on_text(m: types.Message):
    uid = m.from_user.id
    state = WAIT.get(uid)
    if not state or state.get("effect") != "flux":
        return  # реагируем только когда выбран Flux

    prompt = m.text.strip()
    await m.reply("⏳ Генерирую пейзаж по описанию...")
    try:
        out_url = run_epic_landscape_flux(prompt_text=prompt)
        await m.reply_photo(out_url)
    except Exception:
        await m.reply("Не удалось сгенерировать по этому описанию. Попробуй переформулировать.")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling...")
    executor.start_polling(dp, skip_updates=True)
