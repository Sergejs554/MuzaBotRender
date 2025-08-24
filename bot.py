# bot.py — Nature Inspire (Replicate) — CLARITY UPSCALER (со встроенными LoRA)
# env: TELEGRAM_API_TOKEN, REPLICATE_API_TOKEN

import os, logging, tempfile, urllib.request, traceback
from io import BytesIO
from PIL import Image
import replicate
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

# ---------- MODEL REFS ----------
MODEL_CLARITY = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"

# ---------- STATE ----------
WAIT = {}  # user_id -> {'effect': 'nature'}

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

async def send_image_by_url(m: types.Message, url: str):
    """
    Надёжная отправка: качаем и шлём как файл (обход Telegram 'Failed to get http url content').
    Плюс сжатие, если >10MB.
    """
    path = None
    try:
        path = download_to_temp(url)
        path = ensure_photo_size_under_telegram_limit(path)
        await m.reply_photo(InputFile(path))
    finally:
        if path and os.path.exists(path):
            try: os.remove(path)
            except: pass

def pick_first_url(output) -> str:
    """
    У clarity-upscaler реплай — это список blob-объектов.
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

# ---------- PIPELINE: Nature Enhance (Clarity Upscaler, LoRA prompt) ----------
async def run_nature_enhance_with_clarity(file_id: str) -> str:
    """
    Берём TG‑URL → прогоняем через clarity‑upscaler c лорами и «эпик реализмом».
    Настройки — как на твоих скринах (с небольшим апом по шагам до 22 и без даунскейла).
    """
    public_url = await telegram_file_to_public_url(file_id)

    # PROMPT с лорами (влияют на детали/рендер)
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
        "scale_factor": 2,                     # апскейл ×2 (мягче, чем x4; меньше артефактов)
        "dynamic": 6,                          # HDR/динамика как в примере
        "creativity": 0.35,
        "resemblance": 0.6,                    # сохраняем реализм сцены
        "tiling_width": 112,
        "tiling_height": 144,
        "sd_model": "epicrealism_naturalSinRC1VAE.safetensors [84d76a0328]",
        "scheduler": "DPM++ 3M SDE Karras",
        "num_inference_steps": 22,             # +чуть больше шагов
        "seed": 1337,
        "downscaling": False,                  # даунскейл отключён
        # "downscaling_resolution": 768,        # не нужен, раз даунскейл выключен
        "sharpen": 0,                          # sharpening на случай артефактов — 0 по умолчанию
        "handfix": "disabled",
        "output_format": "png",
        # "lora_links": "",                    # не заполняем — в примере лоры встроены через <lora:...>
        # "custom_sd_model": "",               # не используем
    }

    out = replicate.run(MODEL_CLARITY, input=inputs)
    return pick_first_url(out)

# ---------- UI ----------
KB = ReplyKeyboardMarkup(keyboard=[[KeyboardButton("🌿 Nature Enhance")]], resize_keyboard=True)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer("Привет! Nature Enhance = Clarity Upscaler с LoRA и EpicRealism.\nПришли фото — верну усиленный кадр.", reply_markup=KB)

@dp.message_handler(lambda m: m.text == "🌿 Nature Enhance")
async def on_choose(m: types.Message):
    WAIT[m.from_user.id] = {"effect": "nature"}
    await m.answer("Ок! Пришли фото.")

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    state = WAIT.get(m.from_user.id)
    if not state or state.get("effect") != "nature":
        await m.reply("Нажми «🌿 Nature Enhance», затем пришли фото.", reply_markup=KB)
        return

    await m.reply("⏳ Обрабатываю...")
    try:
        url = await run_nature_enhance_with_clarity(m.photo[-1].file_id)
        await send_image_by_url(m, url)
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка nature:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(m.from_user.id, None)

if __name__ == "__main__":
    print(">> Starting polling…")
    executor.start_polling(dp, skip_updates=True)
