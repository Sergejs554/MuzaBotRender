import os
import logging
import numpy as np
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InputFile
from PIL import Image, ImageEnhance, ImageFilter

# ----------------- setup -----------------
logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# ----------------- helpers -----------------
def _pil_to_np(img):
    return np.asarray(img).astype(np.float32) / 255.0

def _np_to_pil(arr):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _screen(a, b):
    # a,b in [0..1]
    return 1.0 - (1.0 - a) * (1.0 - b)

def _softlight(a, b):
    # Photoshop-like softlight approximation
    return np.where(
        b < 0.5,
        a - (1.0 - 2.0*b) * a * (1.0 - a),
        a + (2.0*b - 1.0) * (np.sqrt(a) - a),
    )

def _vignette_mask(h, w, strength=1.6, min_v=0.78):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = 0.5*w, 0.5*h
    r = np.sqrt(((xx - cx)/(0.78*w))**2 + ((yy - cy)/(0.78*h))**2)
    m = 1.0 - np.clip(r**strength, 0.0, 1.0)
    return np.clip(np.maximum(m, min_v), 0.0, 1.0)[..., None]  # HxWx1

# ----------------- effects -----------------
def effect_hdr_sunset(img: Image.Image) -> Image.Image:
    """Базовый HDR: мягкий подъём светов/теней + локальный контраст + насыщенность."""
    img = img.convert("RGB")

    # общий тон: чуть ярче/контрастнее
    img = ImageEnhance.Brightness(img).enhance(1.06)
    img = ImageEnhance.Contrast(img).enhance(1.10)

    # локальный контраст (unsharp)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=120, threshold=2))

    # лёгкий подъём теней через гамму < 1
    arr = _pil_to_np(img)
    gamma = 0.95
    arr = np.clip(arr ** gamma, 0.0, 1.0)

    # умеренная «вибранс»: усиливаем слабонасыщенные больше
    sat_boost = 1.12
    hsv = Image.fromarray((arr*255).astype(np.uint8)).convert("HSV")
    h, s, v = [np.asarray(c, dtype=np.float32)/255.0 for c in hsv.split()]
    s = np.clip(s + (1.0 - s) * (sat_boost - 1.0) * 0.85, 0.0, 1.0)
    hsv = Image.merge("HSV", [Image.fromarray((h*255).astype(np.uint8)),
                              Image.fromarray((s*255).astype(np.uint8)),
                              Image.fromarray((v*255).astype(np.uint8))])
    out = hsv.convert("RGB")
    return out

def effect_golden_hour(img: Image.Image) -> Image.Image:
    """Golden Hour Pop: тёплый софт-лайт + сатурация + мягкая виньетка."""
    img = img.convert("RGB")
    base = _pil_to_np(img)

    # тёплый слой (оранжево-золотой)
    warm = np.ones_like(base)
    warm[..., 0] *= 1.0   # R
    warm[..., 1] *= 0.80  # G
    warm[..., 2] *= 0.55  # B

    # мягкий тёплый тон через softlight
    mix = _softlight(base, warm*0.78)
    mix = np.clip(mix, 0.0, 1.0)

    # лёгкая общая насыщенность
    pil = _np_to_pil(mix)
    pil = ImageEnhance.Color(pil).enhance(1.22)
    pil = ImageEnhance.Contrast(pil).enhance(1.05)

    # виньетка
    arr = _pil_to_np(pil)
    h, w = arr.shape[:2]
    vig = _vignette_mask(h, w, strength=1.7, min_v=0.80)
    arr = np.clip(arr * vig, 0.0, 1.0)

    return _np_to_pil(arr)

def effect_dreamy_mist(img: Image.Image) -> Image.Image:
    """Dreamy Mist: мягкий глоу (screen blend) + лёгкая прохлада + чуть ниже контраст."""
    img = img.convert("RGB")

    # базовая лёгкая «плёнка»
    film = ImageEnhance.Contrast(img).enhance(0.96)
    film = ImageEnhance.Brightness(film).enhance(1.04)

    # bloom / glow
    blur = film.filter(ImageFilter.GaussianBlur(radius=6))
    a = _pil_to_np(film)
    b = _pil_to_np(blur)
    glow = _screen(a, b*0.55)

    # лёгкий холодный оттенок через softlight с голубым
    cool = np.ones_like(glow)
    cool[..., 0] *= 0.80
    cool[..., 1] *= 0.90
    cool[..., 2] *= 1.0
    glow = _softlight(glow, cool*0.65)
    glow = np.clip(glow, 0.0, 1.0)

    # финальный тон
    out = _np_to_pil(glow).filter(ImageFilter.UnsharpMask(radius=0, percent=0, threshold=0))
    return out

# ----------------- UI -----------------
user_mode = {}  # user_id -> 'hdr' | 'golden' | 'mist'

def _kb():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("🌅 HDR Sunset", "🌞 Golden Hour", "🌫️ Dreamy Mist")
    return kb

@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    await message.answer(
        "🌿 Nature Inspire — выбери стиль и пришли фото.\n"
        "FaceCore™ edition (естественно и без «пластмассы»).",
        reply_markup=_kb()
    )

@dp.message_handler(lambda m: m.text in ["🌅 HDR Sunset", "🌞 Golden Hour", "🌫️ Dreamy Mist"])
async def choose_mode(message: types.Message):
    mapping = {
        "🌅 HDR Sunset": "hdr",
        "🌞 Golden Hour": "golden",
        "🌫️ Dreamy Mist": "mist",
    }
    user_mode[message.from_user.id] = mapping[message.text]
    await message.reply("Кинь фото — сделаю магию ✨", reply_markup=_kb())

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    mode = user_mode.get(message.from_user.id)
    if not mode:
        await message.reply("Сначала выбери стиль на клавиатуре ниже.", reply_markup=_kb())
        return

    # сохранить вход
    photo = message.photo[-1]
    inp = f"in_{message.from_user.id}.jpg"
    outp = f"out_{message.from_user.id}.jpg"
    await photo.download(destination_file=inp)

    try:
        img = Image.open(inp)

        if mode == "hdr":
            res = effect_hdr_sunset(img)
            caption = "🌅 HDR Sunset — объём и сочные цвета."
        elif mode == "golden":
            res = effect_golden_hour(img)
            caption = "🌞 Golden Hour — тёплый «золотой час» + виньетка."
        else:
            res = effect_dreamy_mist(img)
            caption = "🌫️ Dreamy Mist — мягкий глоу и кинопрохлада."

        res.save(outp, quality=95)
        await message.answer_photo(InputFile(outp), caption=caption, reply_markup=_kb())

    except Exception as e:
        logging.exception("process error: %s", e)
        await message.reply(f"Ошибка обработки: {e}")
    finally:
        for p in (inp, outp):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
