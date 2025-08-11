import os
import logging
from io import BytesIO

import cv2
import numpy as np
import mediapipe as mp

from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InputFile
from aiogram.utils import executor

API_TOKEN = os.getenv("API_TOKEN")

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Храним выбранный эффект по user_id
user_state = {}

# ---------- МЕНЮ ----------
main_menu = ReplyKeyboardMarkup(resize_keyboard=True)
main_menu.add(KeyboardButton("📸 Make It Special"), KeyboardButton("🌿 Nature Inspire"))

make_it_special_menu = ReplyKeyboardMarkup(resize_keyboard=True)
make_it_special_menu.add(
    KeyboardButton("💖 InstaShine"),
    KeyboardButton("💎 Beauty Sculpt"),
    KeyboardButton("🧴 Skin Retouch"),
    KeyboardButton("🎨 Artistic Tone"),
    KeyboardButton("💫 Light Balance"),
    KeyboardButton("🧠 SmartAI Correction"),
    KeyboardButton("🔙 Назад в меню"),
)

nature_inspire_menu = ReplyKeyboardMarkup(resize_keyboard=True)
nature_inspire_menu.add(
    KeyboardButton("🌄 HDRI Landscape"),
    KeyboardButton("🌿 Natural Boost"),
    KeyboardButton("🌅 Sunset Amplify"),
    KeyboardButton("🪄 Dream Colors"),
    KeyboardButton("🌈 Dynamic Sky"),
    KeyboardButton("🧠 SmartAI Enhance"),
    KeyboardButton("🔙 Назад в меню"),
)

# ---------- БАЗОВЫЕ ХЭНДЛЕРЫ ----------
@dp.message_handler(commands=["start"])
async def send_welcome(message: types.Message):
    await message.answer("Привет, Муза 💫\nВыбери режим редактирования:", reply_markup=main_menu)

@dp.message_handler(lambda m: m.text == "📸 Make It Special")
async def make_it_special(message: types.Message):
    await message.answer("Выбрано: AI Face Magic 💖", reply_markup=make_it_special_menu)

@dp.message_handler(lambda m: m.text == "🌿 Nature Inspire")
async def nature_inspire(message: types.Message):
    await message.answer("Выбрано: AI Scene Enhance 🌿", reply_markup=nature_inspire_menu)

@dp.message_handler(lambda m: m.text == "🔙 Назад в меню")
async def back_to_menu(message: types.Message):
    await message.answer("Вы вернулись в главное меню.", reply_markup=main_menu)

# ---------- ПРОМПТЫ ДЛЯ ЭФФЕКТОВ ----------
@dp.message_handler(lambda m: m.text == "💖 InstaShine")
async def insta_shine_prompt(message: types.Message):
    user_state[message.from_user.id] = "insta_shine"
    await message.answer("💖 Пришли фото для InstaShine — топовый Insta‑Glow эффект.")

@dp.message_handler(lambda m: m.text == "💎 Beauty Sculpt")
async def beauty_sculpt_prompt(message: types.Message):
    user_state[message.from_user.id] = "beauty_sculpt"
    await message.answer("💎 Пришли фото для Beauty Sculpt — деликатное AI‑улучшение черт лица.")

@dp.message_handler(lambda m: m.text == "🧴 Skin Retouch")
async def skin_retouch_prompt(message: types.Message):
    user_state[message.from_user.id] = "skin_retouch"
    await message.answer("🧴 Пришли фото для Skin Retouch — мягкая ретушь без «мыла».")

@dp.message_handler(lambda m: m.text == "🎨 Artistic Tone")
async def artistic_tone_prompt(message: types.Message):
    user_state[message.from_user.id] = "art_tone"
    await message.answer("🎨 Пришли фото для Artistic Tone — кинематографический тон.")

@dp.message_handler(lambda m: m.text == "🌄 HDRI Landscape")
async def hdri_landscape_prompt(message: types.Message):
    user_state[message.from_user.id] = "hdri_landscape"
    await message.answer("🌄 Пришли пейзаж — сделаю микро‑HDR с аккуратной деталью.")

# ---------- УТИЛИТЫ ДЛЯ AI‑АНАЛИЗА ----------
mp_face = mp.solutions.face_mesh

def face_hull_mask(img_bgr):
    """Овал лица по всем точкам FaceMesh (выпуклая оболочка), края слегка подрезаем."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        res = fm.process(img_rgb)

    mask = np.zeros((h, w), dtype=np.uint8)
    if not res.multi_face_landmarks:
        return mask

    pts = np.array(
        [[int(lm.x * w), int(lm.y * h)] for lm in res.multi_face_landmarks[0].landmark],
        dtype=np.int32
    )
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    return mask

def feather(mask, sigma=7):
    return cv2.GaussianBlur(mask, (0, 0), sigma) if sigma > 0 else mask

def blend(src, dst, mask, feather_sigma=8):
    m = feather(mask, feather_sigma).astype(np.float32) / 255.0
    m = m[..., None]
    return (src.astype(np.float32)*(1-m) + dst.astype(np.float32)*m).clip(0,255).astype(np.uint8)

def s_curve(img, strength=0.25):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    x = np.arange(256, dtype=np.float32) / 255.0
    y = 1/(1+np.exp(-(x-0.5)*8*strength))
    y = (y - y.min()) / (y.max()-y.min())
    lut = (y*255).clip(0,255).astype(np.uint8)
    L2 = cv2.LUT(L, lut)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

# --- AI‑Glow для InstaShine (с анализом бликов + лица) ---
def luminance_L(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    return L

def adaptive_bloom(img_bgr, face_mask):
    L = luminance_L(img_bgr)
    p = np.percentile(L[L > 0], 85 if L.mean() > 120 else 80)
    highlight = (L >= p).astype(np.uint8) * 255

    if face_mask is not None and face_mask.sum() > 0:
        mask = cv2.bitwise_and(highlight, face_mask)
    else:
        mask = highlight

    big_blur = cv2.GaussianBlur(img_bgr, (0, 0), 9)
    bloom = cv2.addWeighted(img_bgr, 0.60, big_blur, 0.40, 6)

    avgL = L.mean()
    if avgL > 150:   strength = 0.22
    elif avgL > 120: strength = 0.28
    elif avgL > 90:  strength = 0.33
    else:            strength = 0.38

    mixed = blend(img_bgr, cv2.addWeighted(img_bgr, 1-strength, bloom, strength, 0), mask, feather_sigma=10)
    return mixed

# --- Вспомогательное для Beauty Sculpt ---
def ring_from_mask(mask, inner=18, outer=30):
    """Контурное кольцо вокруг овала лица (для мягкого «скульпта» по периметру)."""
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer, outer))
    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner, inner))
    dil = cv2.dilate(mask, k_out, iterations=1)
    ero = cv2.erode(mask, k_in, iterations=1)
    ring = cv2.subtract(dil, ero)
    return ring

def unsharp(img, amount=0.6, sigma=1.2):
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    return cv2.addWeighted(img, 1+amount, blur, -amount, 0)

# ---------- ОБРАБОТКА ФОТО ----------
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    state = user_state.get(user_id)

    if not state:
        await message.answer("Пожалуйста, сначала выбери эффект из меню.")
        return

    photo = message.photo[-1]
    f = await photo.download(destination=BytesIO())
    f.seek(0)
    np_arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        if state == "insta_shine":
            # AI‑анализ лица + адаптивный glow по бликам
            face_mask = face_hull_mask(img)
            base = img.copy()
            if face_mask.sum() > 0:
                pre = cv2.bilateralFilter(base, 7, 80, 80)
                base = blend(base, pre, face_mask, feather_sigma=6)

            glow = adaptive_bloom(base, face_mask if face_mask.sum() > 0 else None)
            out = s_curve(glow, strength=0.18)

            path = f"insta_shine_{user_id}.jpg"
            cv2.imwrite(path, out)
            await message.answer_photo(InputFile(path), caption="💖 InstaShine готов!")
            os.remove(path)

        elif state == "beauty_sculpt":
            # === Топовый Beauty Sculpt с расширенным AI‑анализом ===
            face_mask = face_hull_mask(img)
            if face_mask.sum() == 0:
                await message.answer("👀 Лицо не найдено — пришли фото фронтально, с хорошим светом.")
            else:
                # 1) Мягкое выравнивание кожи внутри лица (edge‑preserving)
                smooth = cv2.bilateralFilter(img, 13, 140, 140)
                skin_base = blend(img, smooth, face_mask, feather_sigma=8)

                # 2) Контроль света: работаем в LAB, локально выравниваем L (анти‑пересвет/анти‑провал)
                lab = cv2.cvtColor(skin_base, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(lab)
                illum = cv2.GaussianBlur(L, (0,0), 21)                   # оценка освещения
                L_corr = cv2.addWeighted(L, 0.85, illum, -0.15, 10)      # выравниваем экспозицию
                L_corr = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)).apply(L_corr)  # микро‑контраст без клипа
                sculpt_light = cv2.cvtColor(cv2.merge([L_corr, A, B]), cv2.COLOR_LAB2BGR)

                # 3) Контурный «скульпт»: лёгкая резкость по периметру овала + нос/челюсть по кольцу
                ring = ring_from_mask(face_mask, inner=16, outer=28)
                sharp_face = unsharp(sculpt_light, amount=0.45, sigma=1.0)
                sculpted = blend(sculpt_light, sharp_face, ring, feather_sigma=12)

                # 4) Тонкая тональная пластика: слегка поджать хайлайты, приподнять тени внутри лица
                L2 = luminance_L(sculpted).astype(np.float32)
                p_hi = np.percentile(L2[face_mask>0], 85)
                p_lo = np.percentile(L2[face_mask>0], 30)
                hi_mask = ((L2 >= p_hi) & (face_mask>0)).astype(np.uint8)*255
                lo_mask = ((L2 <= p_lo) & (face_mask>0)).astype(np.uint8)*255

                # хайлайты чуть вниз (чтобы не «пластик»), тени — слегка вверх
                down_hi = cv2.convertScaleAbs(sculpted, alpha=0.98, beta=-3)
                up_lo   = cv2.convertScaleAbs(sculpted, alpha=1.03, beta=6)
                sculpted = blend(sculpted, down_hi, hi_mask, feather_sigma=10)
                sculpted = blend(sculpted, up_lo,  lo_mask, feather_sigma=10)

                # 5) Финальная S‑кривая очень деликатно
                out = s_curve(sculpted, strength=0.16)

                path = f"beauty_sculpt_{user_id}.jpg"
                cv2.imwrite(path, out)
                await message.answer_photo(InputFile(path), caption="💎 Beauty Sculpt — объём, свет и чистая кожа без пересвета.")
                os.remove(path)

        elif state == "skin_retouch":
            # Как было — не трогаю
            mask = face_hull_mask(img)
            if mask.sum() == 0:
                await message.answer("👀 Лицо не найдено — пришли фото фронтально, с хорошим светом.")
            else:
                smoothed = cv2.bilateralFilter(img, 9, 120, 120)
                high = cv2.subtract(img, cv2.GaussianBlur(img, (0,0), 3))
                detail_keep = cv2.addWeighted(smoothed, 1.0, high, 0.15, 0)
                out = blend(img, detail_keep, mask, feather_sigma=9)
                path = f"skin_retouch_{user_id}.jpg"
                cv2.imwrite(path, out)
                await message.answer_photo(InputFile(path), caption="🧴 Готово! Мягкая ретушь без «мыла».")
                os.remove(path)

        elif state == "art_tone":
            # Как было — не трогаю
            toned = s_curve(img, strength=0.3)
            wb = toned.astype(np.int16)
            wb[..., 2] = np.clip(wb[..., 2] + 12, 0, 255)   # +R
            wb[..., 0] = np.clip(wb[..., 0] - 8, 0, 255)    # -B
            wb = wb.astype(np.uint8)
            h, w = wb.shape[:2]
            y, x = np.ogrid[:h, :w]
            cy, cx = h/2, w/2
            vign = ((x-cx)**2 + (y-cy)**2) / (max(h,w)**2)
            vign = (1 - 0.35*vign).clip(0.65, 1.0).astype(np.float32)
            vign = cv2.GaussianBlur(vign, (0,0), max(h,w)*0.02)
            out = (wb.astype(np.float32) * vign[..., None]).clip(0,255).astype(np.uint8)
            path = f"art_tone_{user_id}.jpg"
            cv2.imwrite(path, out)
            await message.answer_photo(InputFile(path), caption="🎨 Готово! Кинематографический тон применён.")
            os.remove(path)

        elif state == "hdri_landscape":
            # Как было — не трогаю
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L,A,B = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            L2 = clahe.apply(L)
            base = cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2BGR)
            detail = cv2.detailEnhance(base, sigma_s=12, sigma_r=0.15)
            out = cv2.addWeighted(base, 0.6, detail, 0.4, 0)
            path = f"hdri_{user_id}.jpg"
            cv2.imwrite(path, out)
            await message.answer_photo(InputFile(path), caption="🌄 Готово! HDRI‑укрепление сцены выполнено.")
            os.remove(path)

        else:
            await message.answer("Эффект ещё в разработке. Выбери другой.")

    finally:
        user_state.pop(user_id, None)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
