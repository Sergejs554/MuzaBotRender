import os
import logging
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InputFile
from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

user_mode = {}

@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["📸 Make It Special", "🌿 Nature"]
    keyboard.add(*buttons)
    await message.answer("Привет! Выбери режим обработки:", reply_markup=keyboard)

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
    file_path = f"input_{message.from_user.id}.jpg"
    result_path = f"output_{message.from_user.id}.jpg"

    await photo.download(destination_file=file_path)

    # Простейшая обработка в зависимости от режима
    image = Image.open(file_path)

    if mode == "face":
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
    else:  # nature
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.8)

    image.save(result_path)

    await bot.send_photo(chat_id=message.chat.id, photo=InputFile(result_path), caption="Вот результат!")
    os.remove(file_path)
    os.remove(result_path)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
