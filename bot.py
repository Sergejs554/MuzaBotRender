import os
import logging
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InputFile

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
    await message.answer("Привет! Выбери режим обработки:", reply_markup=keyboard)

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
    file_path = f"input_{message.from_user.id}.jpg"
    await photo.download(destination_file=file_path)

    result_path = f"output_{message.from_user.id}.jpg"
    # Обработка: подмена на заглушку (настоящую обработку подключим позже)
    if mode == "face":
        placeholder = "samples/face_example.jpg"
    else:
        placeholder = "samples/nature_example.jpg"

    await bot.send_photo(chat_id=message.chat.id, photo=InputFile(placeholder), caption="Вот результат!")
    os.remove(file_path)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
