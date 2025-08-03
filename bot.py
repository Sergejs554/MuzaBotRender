import telebot
import os

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет, Вика, Муза! Пришли мне любое фото, и ты увидишь магию.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, "🧚‍♀️ Обрабатываю твоё фото... (демо-режим)")

print("Бот запущен. Ожидает команды...")

bot.infinity_polling()
