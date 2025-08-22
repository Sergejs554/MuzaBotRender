# bot.py — Nature Inspire (Replicate)

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
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ----- MODELS -----
MODEL_FLUX    = "black-forest-labs/flux-1.1-pro"
MODEL_REFINER = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_ESRGAN  = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
MODEL_SWINIR  = "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"

# ----- STATE -----
WAIT = {}  # user_id -> {'effect': ...}

def tg_file_url(file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

def pick_url(output):
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
    🌿 Magic Image Refiner (со strength) -> ESRGAN x2
    """
    base_prompt = "natural color balance, clean details, no artifacts, no extra objects"
    # 1) Пытаемся с strength
    try:
        ref_out = replicate.run(
            MODEL_REFINER,
            input={
                "image": public_url,
                "prompt": base_prompt,
                "strength": 0.85  # если поле не поддерживается, поймаем 422 и откатимся
            }
        )
    except Exception:
        # 2) Фоллбек без strength (совместимость с другими сборками)
        ref_out = replicate.run(
            MODEL_REFINER,
            input={"image": public_url, "prompt": base_prompt}
        )
    ref_url = pick_url(ref_out)

    # 3) ESRGAN x2 — всегда
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": ref_url, "scale": 2})
    return pick_url(esr_out)

def run_epic_landscape_flux(prompt_text: str) -> str:
    if not prompt_text or not prompt_text.strip():
        prompt_text = (
            "epic panoramic landscape, dramatic sky, volumetric light, ultra-detailed mountains, "
            "lush forests, cinematic composition, award-winning nature photography"
        )
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    return pick_url(flux_out)

def run_ultra_hdr(public_url: str, hint_caption: str = "") -> str:
    """
    🏞 FLUX (по тексту-наводке) -> ESRGAN x2
    Если подпись пустая — автопромпт, чтобы не падало и не уходило в «пришли другую фотку».
    """
    prompt_text = hint_caption.strip() if hint_caption else "Ultra HDR, realistic photo"
    flux_out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    flux_url = pick_url(flux_out)
    esr_out = replicate.run(MODEL_ESRGAN, input={"image": flux_url, "scale": 2})
    return pick_url(esr_out)

def run_clean_restore(public_url: str) -> str:
    swin_out = replicate.run(
        MODEL_SWINIR,
        input={"image": public_url, "jpeg": "40", "noise": "15"}
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
        "Выбери режим ниже. Для Flux можно прислать только текст (промпт).",
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
        await m.answer("Пришли описание сцены (можно без фото) — сгенерю эпик‑ландшафт.")
    elif "Ultra HDR" in m.text:
        WAIT[uid] = {"effect": "hdr"}
        await m.answer("Пришли фото. Можно приложить подпись — усилю сцену в стиле HDR.")
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
        tg_file = await bot.get_file(m.photo[-1].file_id)
        public_url = tg_file_url(tg_file.file_path)

        if effect == "nature":
            out_url = run_nature_enhance(public_url)
        elif effect == "flux":
            out_url = run_epic_landscape_flux(prompt_text=caption)
        elif effect == "hdr":
            out_url = run_ultra_hdr(public_url, hint_caption=caption)
        elif effect == "clean":
            out_url = run_clean_restore(public_url)
        else:
            raise RuntimeError("Unknown effect")

        await m.reply_photo(out_url)

    except Exception:
        await m.reply("Не удалось обработать фото. Попробуй другую фотографию или другую подпись.")
    finally:
        WAIT.pop(uid, None)

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
        await m.reply("Не удалось сгенерировать по этому описанию. Попробуй переформулировать.")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling...")
    executor.start_polling(dp, skip_updates=True)
