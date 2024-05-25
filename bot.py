import json
import telebot
import configparser
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configs
config = configparser.ConfigParser()
config.read('config.ini')

token = config['bot']['ACCESS_TOKEN']
bot = telebot.TeleBot(token)

# Models and tokens
with open('./resources/word_index.json', 'r', encoding='utf-8') as f:
    loaded_word_index = json.load(f)
tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.word_index = loaded_word_index

loaded_model = load_model('./resources/model.h5')


def predict_toxicity(text: str) -> str:
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=200)

    prediction = loaded_model.predict(text_pad)
    toxicity_probability = prediction[0][0]
    return f"Степень токсичности текста: {toxicity_probability * 100:.2f}%"


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, text=
    "Привет! Я бот, который умеет определять токсичность текста! "
    "Напиши мне текст и я помогу тебе определить токсичен ли он или нет!")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    response = predict_toxicity(message.text)
    bot.send_message(message.chat.id, response)


bot.polling()
