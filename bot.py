# bot.py — Nature Inspire: ProShot Lens (local) & ProShot + Clarity (+ESRGAN)
# env: TELEGRAM_API_TOKEN, REPLICATE_API_TOKEN

import os, logging, tempfile, urllib.request, traceback, math
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops
import replicate
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InputFile
from aiogram.utils import executor

logging.basicConfig(level=logging.INFO)

# ---------- TOKENS ----------
API_TOKEN  = os.getenv("TELEGRAM_API_TOKEN")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("TELEGRAM_API_TOKEN missing")
if not REPL_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN missing")
os.environ["REPLICATE_API_TOKEN"] = REPL_TOKEN

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ---------- MODELS ----------
MODEL_CLARITY = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"
MODEL_ESRGAN  = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"

# ---------- TUNABLES ----------
INPUT_MAX_SIDE        = 1536
FINAL_TELEGRAM_LIMIT  = 10 * 1024 * 1024
ESRGAN_MAX_INPUT_PIXELS = 2_000_000
UPSCALE_AFTER          = True
UPSCALE_SCALE          = 2

# Clarity knobs
CLARITY_SCALE_FACTOR     = 2
CLARITY_DYNAMIC          = 5.0
CLARITY_CREATIVITY       = 0.22
CLARITY_RESEMBLANCE      = 0.72
CLARITY_TILING_W         = 112
CLARITY_TILING_H         = 144
CLARITY_STEPS            = 20
CLARITY_SD_MODEL         = "juggernaut_reborn.safetensors [338b85bc4f]"
CLARITY_SCHEDULER        = "DPM++ 3M SDE Karras"
CLARITY_MORE_DETAILS_LORA= 0.45
CLARITY_RENDER_LORA      = 0.9

# ProShot Lens knobs (меняй цифрами)
PRO_CONTRAST    = 1.10   # глобальный контраст
PRO_FILMIC_CURVE= 0.15   # S-кривая (0..0.35)
PRO_WARMTH      = 1.06   # тёплый баланс (R↑, B↓)
PRO_VIBRANCE    = 1.12   # «вибранс» (щадящая насыщенность)
PRO_HALATION    = 0.18   # блум хайлайтов (0..0.35)
PRO_GRAIN       = 0.06   # плёночное зерно (0..0.15)
PRO_SHARP       = 110    # Unsharp percent (0..180)
PRO_BLEND       = 0.15   # смешивание с оригиналом (0..0.35) — «человечность»

# ---------- STATE ----------
WAIT = {}  # user_id -> {'effect': 'pro'|'clarity_pro', 'strength': 'low|med|high'}

# ---------- HELPERS ----------
def tg_public_url(file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

async def download_tg_photo(file_id: str) -> str:
    tg_file = await bot.get_file(file_id)
    url = tg_public_url(tg_file.file_path)
    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def resize_inplace(path: str, max_side: int):
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        img.save(path, "JPEG", quality=95, optimize=True)
    except Exception:
        pass

def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def ensure_size_under_telegram_limit(path: str, max_bytes: int = FINAL_TELEGRAM_LIMIT) -> str:
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
            os.remove(tmp); q -= 8
        fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        img.save(tmp, "JPEG", quality=max(q, 40), optimize=True)
        try: os.remove(path)
        except: pass
        return tmp
    except Exception:
        return path

def pick_first_url(output) -> str:
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

# ---------- ESRGAN (safe input) ----------
def esrgan_upscale_path(path: str, scale: int = 2) -> str:
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    w, h = im.size
    if w*h > ESRGAN_MAX_INPUT_PIXELS:
        k = (ESRGAN_MAX_INPUT_PIXELS / (w*h)) ** 0.5
        nw, nh = max(256, int(w*k)), max(256, int(h*k))
        im = im.resize((nw, nh), Image.LANCZOS)
        fd, safe_in = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        im.save(safe_in, "JPEG", quality=95, optimize=True)
        in_path = safe_in
    else:
        in_path = path

    with open(in_path, "rb") as bf:
        out = replicate.run(MODEL_ESRGAN, input={"image": bf, "scale": scale})
    url = pick_first_url(out)
    tmp = download_to_temp(url)
    if in_path != path:
        try: os.remove(in_path)
        except: pass
    return tmp

# ---------- ProShot Lens (локально, «плёночный про-объектив») ----------
def _apply_filmic_curve(arr, s=0.15):
    # простая S-кривая: mix линейного и smoothstep
    x = np.clip(arr, 0, 1)
    y = x*(1-s) + (3*x*x - 2*x*x*x)*s
    return np.clip(y, 0, 1)

def _apply_warmth(arr, k=1.06):
    # тёплый баланс: R↑, B↓ (деликатно, через матрицу)
    r,g,b = arr[...,0], arr[...,1], arr[...,2]
    r = np.clip(r * k, 0, 1)
    b = np.clip(b / k, 0, 1)
    return np.stack([r,g,b], axis=-1)

def _vibrance(img: Image.Image, amount: float) -> Image.Image:
    # «вибранс»: усиливаем мало-насыщенные пиксели больше, чем уже насыщенные
    arr = np.asarray(img).astype(np.float32)/255.0
    mx = arr.max(axis=-1, keepdims=True)
    mn = arr.min(axis=-1, keepdims=True)
    sat = (mx - mn)
    w = 1.0 - sat                              # слабосат. пикселям — больший вес
    sat_boost = 1.0 + (amount-1.0)*w
    mean = arr.mean(axis=-1, keepdims=True)
    arr = mean + (arr-mean)*sat_boost
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8))

def _halation(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0: return img
    blur = img.filter(ImageFilter.GaussianBlur(radius=2.0 + 6.0*strength))
    # выделяем хайлайты
    g = blur.convert("L")
    g = g.point(lambda v: int(max(0, v-200) * (255/55)))  # порог ~200
    g = g.filter(ImageFilter.GaussianBlur(radius=1.0 + 3.0*strength))
    glow = ImageChops.multiply(blur, Image.merge("RGB",(g,g,g)))
    return Image.blend(img, glow, strength*0.6)

def _add_grain(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0: return img
    w,h = img.size
    noise = np.random.normal(0, strength*25.0, (h,w,1)).astype(np.float32)/255.0
    arr = np.asarray(img).astype(np.float32)/255.0
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8))

def proshot_enhance_path(orig_path: str,
                         contrast=PRO_CONTRAST,
                         filmic=PRO_FILMIC_CURVE,
                         warmth=PRO_WARMTH,
                         vibrance=PRO_VIBRANCE,
                         halation=PRO_HALATION,
                         grain=PRO_GRAIN,
                         sharp_percent=PRO_SHARP,
                         blend=PRO_BLEND) -> str:
    base = Image.open(orig_path).convert("RGB")
    base = ImageOps.exif_transpose(base)

    # 1) лёгкий контраст + S-кривая
    im = ImageEnhance.Contrast(base).enhance(contrast)
    arr = np.asarray(im).astype(np.float32)/255.0
    arr = _apply_filmic_curve(arr, s=filmic)
    arr = _apply_warmth(arr, k=warmth)
    im = Image.fromarray((arr*255).astype(np.uint8))

    # 2) vibrance (щадящая насыщенность)
    im = _vibrance(im, vibrance)

    # 3) halation/bloom по хайлайтам
    im = _halation(im, halation)

    # 4) микрошарп
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharp_percent), threshold=2))

    # 5) плёночный grain
    im = _add_grain(im, grain)

    # 6) «человечность»: смешаем с оригиналом
    out = Image.blend(base, im, max(0.0, min(1.0, blend)))

    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    out.save(path, "JPEG", quality=95, optimize=True)
    return path

# ---------- PIPELINES ----------
async def run_pro_lens_only(file_id: str) -> str:
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)
    try:
        pro_path = proshot_enhance_path(local_in)
    finally:
        try: os.remove(local_in)
        except: pass

    if UPSCALE_AFTER:
        up = esrgan_upscale_path(pro_path, scale=UPSCALE_SCALE)
        try: os.remove(pro_path)
        except: pass
        pro_path = up
    return pro_path

async def run_pro_lens_with_clarity(file_id: str) -> str:
    # 1) качаем и ресайзим
    local_in = await download_tg_photo(file_id)
    resize_inplace(local_in, INPUT_MAX_SIDE)

    # 2) Clarity (мягкие настройки, ближе к реальности)
    prompt_text = (
        "masterpiece, best quality, highres,\n"
        f"<lora:more_details:{CLARITY_MORE_DETAILS_LORA}>\n"
        f"<lora:SDXLrender_v2.0:{CLARITY_RENDER_LORA}>"
    )
    negative = "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"
    with open(local_in, "rb") as f:
        out = replicate.run(MODEL_CLARITY, input={
            "image": f,
            "prompt": prompt_text,
            "negative_prompt": negative,
            "scale_factor": CLARITY_SCALE_FACTOR,
            "dynamic": CLARITY_DYNAMIC,
            "creativity": CLARITY_CREATIVITY,
            "resemblance": CLARITY_RESEMBLANCE,
            "tiling_width": CLARITY_TILING_W,
            "tiling_height": CLARITY_TILING_H,
            "sd_model": CLARITY_SD_MODEL,
            "scheduler": CLARITY_SCHEDULER,
            "num_inference_steps": CLARITY_STEPS,
            "seed": 1337,
            "downscaling": False,
            "sharpen": 0,
            "handfix": "disabled",
            "output_format": "png",
        })
    try: os.remove(local_in)
    except: pass

    cl_url  = pick_first_url(out)
    cl_path = download_to_temp(cl_url)

    # 3) ProShot поверх Clarity
    try:
        pro_path = proshot_enhance_path(cl_path)
    finally:
        try: os.remove(cl_path)
        except: pass

    # 4) (опц.) апскейл
    if UPSCALE_AFTER:
        up = esrgan_upscale_path(pro_path, scale=UPSCALE_SCALE)
        try: os.remove(pro_path)
        except: pass
        pro_path = up
    return pro_path

# ---------- UI ----------
KB_MAIN = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("📸 ProShot Lens")],
        [KeyboardButton("📸 ProShot + Clarity")],
        # HDR убран из меню — добавим позже отдельной кнопкой
    ],
    resize_keyboard=True
)

@dp.message_handler(commands=["start"])
async def on_start(m: types.Message):
    await m.answer(
        "Nature Inspire 📸\n"
        "• ProShot Lens — плёночный про-объектив (натурально, без пластика)\n"
        "• ProShot + Clarity — сначала Clarity, затем ProShot (+опц. апскейл)\n"
        "Пришли фото после выбора режима.",
        reply_markup=KB_MAIN
    )

@dp.message_handler(lambda m: m.text in ["📸 ProShot Lens", "📸 ProShot + Clarity"])
async def on_choose_mode(m: types.Message):
    uid = m.from_user.id
    WAIT[uid] = {"effect": "pro" if "Lens" in m.text else "clarity_pro"}
    await m.answer("Ок! Пришли фото.")

@dp.message_handler(content_types=["photo"])
async def on_photo(m: types.Message):
    uid = m.from_user.id
    st = WAIT.get(uid)
    if not st or st.get("effect") not in ["pro", "clarity_pro"]:
        await m.reply("Сначала выбери режим ⬇️", reply_markup=KB_MAIN)
        return

    await m.reply("⏳ Обрабатываю...")
    try:
        if st["effect"] == "pro":
            out_path = await run_pro_lens_only(m.photo[-1].file_id)
        else:
            out_path = await run_pro_lens_with_clarity(m.photo[-1].file_id)

        safe = ensure_size_under_telegram_limit(out_path)
        await m.reply_photo(InputFile(safe))
        try:
            if os.path.exists(out_path): os.remove(out_path)
            if safe != out_path and os.path.exists(safe): os.remove(safe)
        except: pass
    except Exception:
        tb = traceback.format_exc(limit=20)
        await m.reply(f"🔥 Ошибка Nature Inspire:\n```\n{tb}\n```", parse_mode="Markdown")
    finally:
        WAIT.pop(uid, None)

if __name__ == "__main__":
    print(">> Starting polling…")
    executor.start_polling(dp, skip_updates=True)
