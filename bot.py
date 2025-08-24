# bot.py — Nature Inspire (Replicate) — SOLO Clarity-Upscaler (flagship)
# env: TELEGRAM_API_TOKEN, REPLICATE_API_TOKEN
# pip: aiogram==2.25.1 pillow replicate

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
MODEL_CLARITY = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"
MODEL_FLUX    = "black-forest-labs/flux-1.1-pro"  # оставляем как было
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
            return str(url_attr() if callable(url_attr) else (url_attr or o0))
        url_attr = getattr(output, "url", None)
        return str(url_attr() if callable(url_attr) else (url_attr or output))
    except Exception:
        return str(output)

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def ensure_photo_size_under_telegram_limit(path: str, max_bytes: int = 10 * 1024 * 1024) -> str:
    """Если файл >10MB — пережимаем JPEG до лимита."""
    try:
        if os.path.getsize(path) <= max_bytes:
            return path
        img = Image.open(path).convert("RGB")
        q = 92
        for _ in range(10):
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(tmp_fd)
            img.save(tmp_path, "JPEG", quality=q, optimize=True)
            if os.path.getsize(tmp_path) <= max_bytes:
                try: os.remove(path)
                except: pass
                return tmp_path
            os.remove(tmp_path)
            q -= 8
        final_fd, final_path = tempfile.mkstemp(suffix=".jpg")
        os.close(final_fd)
        img.save(final_path, "JPEG", quality=max(q, 40), optimize=True)
        try: os.remove(path)
        except: pass
        return final_path
    except Exception:
        return path

async def send_image_by_url(m: types.Message, url: str):
    """Качаем результат и шлём как файл (обход ‘Failed to get http url content’) + соблюдаем 10MB."""
    path = None
    try:
        path = download_to_temp(url)
        path = ensure_photo_size_under_telegram_limit(path)
        await m.reply_photo(InputFile(path))
    finally:
        if path and os.path.exists(path):
            try: os.remove(path)
            except: pass

async def download_and_resize_input(file_id: str, max_side: int = 2048) -> str:
    """
    Скачиваем из TG и мягко уменьшаем до безопасного размера по длинной стороне
    (чтобы стабильно проходило на модели и держать разумный вес).
    """
    public_url = await telegram_file_to_public_url(file_id)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(public_url, path)

    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            img.save(path, "JPEG", quality=95, optimize=True)
    except Exception:
        pass
    return path

# ===================== PIPELINES =====================

async def run_nature_clarity(file_id: str) -> str:
    """
    SOLO Clarity-Upscaler: акцент на микродеталях + цвет, без мыла и без «пластика».
    Схема: TG → безопасный ресайз → clarity-upscaler (x4).
    """
    tmp_path = await download_and_resize_input(file_id, max_side=2048)
    try:
        # Фокусный промпт — короткий и точный (микродеталь + реализм + HDR баланс)
        prompt = (
            "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>, "
        )
        negative_prompt = (
            "low quality, blurry, watercolor, smudged, noise, artifact, oversaturated, "
            "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"
        )

        with open(tmp_path, "rb") as f:
            out = replicate.run(
                MODEL_CLARITY,
                input = {
                    "image": <URL_ИЛИ_ФАЙЛ>,
                    "prompt": "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>",
                    "negative_prompt": "(worst quality, low quality, normal quality:2) JuggernautNegative-neg",
                    "scale_factor": 2,
                    "dynamic": 6,
                    "creativity": 0.35,
                    "resemblance": 0.6,
                    "scheduler": "DPM++ 2M Karras",
                    "num_inference_steps": 18,
                    "seed": 1337,
                    "tiling_width": 16,
                    "tiling_height": 16,
                    "sd_model": "juggernaut_reborn.safetensors [338b85bc4f]"
                
                    # seed опускаем, пусть будет рандом (меньше повторов)
                }
            )
        return pick_url(out)
    finally:
        try: os.remove(tmp_path)
        except: pass

def run_epic_landscape_flux(prompt_text: str) -> str:
    if not prompt_text or not prompt_text.strip():
        prompt_text = (
            "epic panoramic landscape, dramatic sky, volumetric light, ultra-detailed mountains, "
            "lush forests, cinematic composition, award-winning nature photography"
        )
    out = replicate.run(MODEL_FLUX, input={"prompt": prompt_text, "prompt_upsampling": True})
    return pick_url(out)

def run_clean_restore(public_url: str) -> str:
    swin_out = replicate.run(MODEL_SWINIR, input={"image": public_url, "jpeg": "40", "noise": "15"})
    return pick_url(swin_out)

# ===================== UI / HANDLERS =====================

KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🌿 Nature Enhance")],
        [KeyboardButton("🌄 Epic Landscape Flux")],
        [KeyboardButton("📸 Clean Restore")],
    ],
    resize_keyboard=True
)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer(
        "Привет ✨ Природу усилим по флагманской схеме (Clarity‑Upscaler x4).\n"
        "Выбери режим ниже и пришли фото (для Flux можно просто текст).",
        reply_markup=KB
    )

@dp.message_handler(lambda m: m.text in ["🌿 Nature Enhance", "🌄 Epic Landscape Flux", "📸 Clean Restore"])
async def on_choose(m: types.Message):
    uid = m.from_user.id
    if "Nature Enhance" in m.text:
        WAIT[uid] = {"effect": "nature"}
        await m.answer("Ок! Пришли фото. ⛰️🌿")
    elif "Epic Landscape Flux" in m.text:
        WAIT[uid] = {"effect": "flux"}
        await m.answer("Пришли подпись-описание пейзажа (или просто текст без фото) — сгенерю кадр.")
    elif "Clean Restore" in m.text:
        WAIT[uid] = {"effect": "clean"}
        await m.answer("Пришли фото. Уберу шум/мыло аккуратно.")

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
        if effect == "nature":
            out_url = await run_nature_clarity(m.photo[-1].file_id)
            await send_image_by_url(m, out_url)
        elif effect == "flux":
            out_url = run_epic_landscape_flux(prompt_text=caption)
            await send_image_by_url(m, out_url)
        elif effect == "clean":
            public_url = await telegram_file_to_public_url(m.photo[-1].file_id)
            out_url = run_clean_restore(public_url)
            await send_image_by_url(m, out_url)
        else:
            await m.reply("Неизвестный режим.")
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка {effect}:\n```\n{tb}\n```", parse_mode="Markdown")
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
        await send_image_by_url(m, out_url)
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка flux:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling...")
    executor.start_polling(dp, skip_updates=True)
