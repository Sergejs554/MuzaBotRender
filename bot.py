# bot.py — Nature Inspire (Replicate) — Refiner + Swin2SR x4 (флагман)
# env: TELEGRAM_API_TOKEN, REPLICATE_API_TOKEN

import os
import logging
import replicate
import asyncio
import traceback
import urllib.request
import tempfile
from io import BytesIO
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
# FLUX и прочее оставил без изменений, но основной апдейт — Nature Enhance
MODEL_FLUX      = "black-forest-labs/flux-1.1-pro"
MODEL_REFINER   = "fermatresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"
MODEL_SWIN2SR   = "mv-lab/swin2sr:a01b0512004918ca55d02e554914a9eca63909fa83a29ff0f115c78a7045574f"  # x4 SR

# (если нужно, можно оставить и другие эффекты)
MODEL_ESRGAN    = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
MODEL_SWINIR    = "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"

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
            # у разных SDK .url — свойство или метод
            url_attr = getattr(o0, "url", None)
            if callable(url_attr):
                return str(url_attr())
            if url_attr:
                return str(url_attr)
            return str(o0)
        url_attr = getattr(output, "url", None)
        if callable(url_attr):
            return str(url_attr())
        if url_attr:
            return str(url_attr)
        return str(output)
    except Exception:
        return str(output)

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def ensure_photo_size_under_telegram_limit(path: str, max_bytes: int = 10 * 1024 * 1024) -> str:
    """
    Telegram ограничивает отправку photo ~10MB.
    Если файл больше — пережимаем в JPEG с постепенным снижением качества.
    """
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
                try:
                    os.remove(path)
                except Exception:
                    pass
                return tmp_path
            os.remove(tmp_path)
            q -= 8
        # если не уложились — сохраняем последний вариант с минимальным q
        final_fd, final_path = tempfile.mkstemp(suffix=".jpg")
        os.close(final_fd)
        img.save(final_path, "JPEG", quality=max(q, 40), optimize=True)
        try:
            os.remove(path)
        except Exception:
            pass
        return final_path
    except Exception:
        # если что-то пошло не так — вернём исходный путь (может повезёт)
        return path

async def send_image_by_url(m: types.Message, url: str):
    """
    Надёжная отправка: качаем результат и шлём как файл (обход «Failed to get http url content»).
    Плюс сжатие, если >10MB.
    """
    path = None
    try:
        path = download_to_temp(url)
        path = ensure_photo_size_under_telegram_limit(path)
        await m.reply_photo(InputFile(path))
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

async def download_and_resize_input(file_id: str, max_side: int = 1536) -> str:
    """
    Скачиваем вход из TG и мягко уменьшаем до безопасного размера (по длинной стороне),
    чтобы не упираться в лимиты GPU на стороне Replicate и ускорить обработку.
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

def run_nature_enhance_refiner_only(public_url: str) -> str:
    """
    Лёгкая очистка/баланс от Refiner — без агрессии, чтобы сохранить текстуры.
    """
    ref_inputs = {
        "image": public_url,
        # мягкий промпт: убрать мутность/грязь, не теряя деталей
        "prompt": "subtle clarity, realistic color balance, preserve fine textures, no extra objects, no oversharpen"
    }
    ref_out = replicate.run(MODEL_REFINER, input=ref_inputs)
    return pick_url(ref_out)

def run_swin2sr_x4(public_url: str) -> str:
    """
    Swin2SR супер‑разрешение (x4) — модель по умолчанию выдаёт x4, параметров масштаба не требует.
    """
    sr_out = replicate.run(MODEL_SWIN2SR, input={"image": public_url})
    return pick_url(sr_out)

def run_nature_enhance_pipeline(file_id: str) -> str:
    """
    НОВЫЙ Nature Enhance = (вход TG) → безопасный ресайз → Refiner (лёгкий) → Swin2SR (x4).
    Возвращаем финальный URL.
    """
    # 1) скачали и «обезопасили» размер
    tmp_path = asyncio.get_event_loop().run_until_complete(download_and_resize_input(file_id, 1536))

    try:
        # заливаем локальный файл в Refiner напрямую (SDK сам загрузит файл)
        ref_url = run_nature_enhance_refiner_only(tmp_path)

        # 2) Swin2SR (x4) — главное для детализации
        sr_url = run_swin2sr_x4(ref_url)
        return sr_url
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# (оставшиеся эффекты — без изменений; можно отключить, если мешают)
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
    return pick_url(flux_out)

def run_clean_restore(public_url: str) -> str:
    swin_out = replicate.run(MODEL_SWINIR, input={"image": public_url, "jpeg": "40", "noise": "15"})
    return pick_url(swin_out)

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
        "Привет ✨ Природу усилим по флагманской схеме (Refiner + Swin2SR x4).\n"
        "Выбери режим ниже и пришли фото (для Flux можно просто текст).",
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
            out_url = run_nature_enhance_pipeline(m.photo[-1].file_id)
            await send_image_by_url(m, out_url)
        elif effect == "flux":
            out_url = run_epic_landscape_flux(prompt_text=caption)
            await send_image_by_url(m, out_url)
        elif effect == "hdr":
            out_url = run_ultra_hdr("", hint_caption=caption)
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
