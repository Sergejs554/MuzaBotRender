import os
import logging
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InputFile
from PIL import Image, ImageEnhance

# Логирование
logging.basicConfig(level=logging.INFO)

# Получение токена
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Хэндлер старта
@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["📸 Make It Special", "🌿 Nature"]
    keyboard.add(*buttons)
    await message.answer("Привет, Муза. Пришли любое фото — и ты увидишь магию ✨", reply_markup=keyboard)

# Состояние выбранного режима
user_mode = {}

@dp.message_handler(lambda message: message.text in ["📸 Make It Special", "🌿 Nature"])
async def choose_mode(message: types.Message):
    mode = "face" if message.text == "📸 Make It Special" else "nature"
    user_mode[message.from_user.id] = mode
    await message.reply("Отправь фото для обработки!")

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    mode = user_mode.get(message.from_user.id)
    if not mode:
        await message.reply("Пожалуйста, сначала выбери режим обработки.")
        return

    photo = message.photo[-1]
    input_path = f"input_{message.from_user.id}.jpg"
    output_path = f"output_{message.from_user.id}.jpg"
    await photo.download(destination_file=input_path)

    # Обработка фотографии
    try:
        image = Image.open(input_path)

        if mode == "face":
            # Мягкая и естественная коррекция
            image = image.convert("RGB")
            image = ImageEnhance.Brightness(image).enhance(1.08)
            image = ImageEnhance.Contrast(image).enhance(1.05)
            image = ImageEnhance.Color(image).enhance(1.03)

        elif mode == "nature":
            # Яркая коррекция для природы
            image = image.convert("RGB")
            image = ImageEnhance.Brightness(image).enhance(1.15)
            image = ImageEnhance.Contrast(image).enhance(1.15)
            image = ImageEnhance.Color(image).enhance(1.25)

        image.save(output_path)
        await bot.send_photo(chat_id=message.chat.id, photo=InputFile(output_path), caption="Готово! ✨")

    except Exception as e:
        await message.reply(f"Ошибка при обработке изображения: {e}")

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
