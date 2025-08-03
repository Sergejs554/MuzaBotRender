import os
import logging
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InputFile
from PIL import Image, ImageEnhance

# Логирование
logging.basicConfig(level=logging.INFO)

# Токен бота из переменных окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Состояние выбранного режима
user_mode = {}

# Хэндлер старта
@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["📸 Make It Special", "🌿 Nature"]
    keyboard.add(*buttons)
    await message.answer("Привет! Выбери режим обработки:", reply_markup=keyboard)

# Выбор режима
@dp.message_handler(lambda message: message.text in ["📸 Make It Special", "🌿 Nature"])
async def choose_mode(message: types.Message):
    mode = "face" if message.text == "📸 Make It Special" else "nature"
    user_mode[message.from_user.id] = mode
    await message.reply("Отправь фото для обработки!")

# Обработка фото
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    mode = user_mode.get(message.from_user.id)
    if not mode:
        await message.reply("Сначала выбери режим обработки.")
        return

    photo = message.photo[-1]
    file_path = f"input_{message.from_user.id}.jpg"
    await photo.download(destination_file=file_path)

    result_path = f"output_{message.from_user.id}.jpg"
    try:
        image = Image.open(file_path)

        if mode == "face":
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
        else:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)

        image.save(result_path)
        await message.answer_photo(types.InputFile(result_path), caption="Вот результат!")

    except Exception as e:
        await message.reply("Ошибка при обработке изображения.")
        print(e)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(result_path):
            os.remove(result_path)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
