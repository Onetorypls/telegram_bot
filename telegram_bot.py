!pip install python-telegram-bot --upgrade
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import logging
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

with open('bot_config.json') as f:
  BOT_CONFIG = json.load(f)
del BOT_CONFIG['intents']['price']

texts = []
y = []
for intent in BOT_CONFIG['intents'].keys():
  for example in BOT_CONFIG['intents'][intent]['examples']:
    texts.append(example)
    y.append(intent)

train_texts, test_texts, y_train, y_test = train_test_split(texts, y, random_state=42, test_size=0.2)

def clean(text):
  clean_text = ''
  for ch in text.lower():
    if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя abcdefghijklmnopqrstuvwxyz!,.?':
      clean_text += ch
  return clean_text

vectorizer = CountVectorizer(preprocessor=clean, ngram_range=(1,6), analyzer='char_wb')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

vocab = vectorizer.get_feature_names_out()

clf = RandomForestClassifier(n_estimators=300, min_samples_split=3).fit(X_train, y_train)
clf.score(X_train, y_train), clf.score(X_test, y_test)

def get_intent_by_model(text):
  return clf.predict(vectorizer.transform([text]))[0]

def bot(input_text):
  intent = get_intent_by_model(input_text)
  return random.choice(BOT_CONFIG['intents'][intent]['responses'])

input_text = ''
while input_text != 'stop':
  input_text = input()
  if input_text != 'stop':
    response = bot(input_text)
    print(response)

import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    input_text = update.message.text
    output_text = bot(input_text)
    update.message.reply_text(output_text)


def main() -> None:
    """Start the bot."""
    updater = Updater("5024309089:AAF6s7xIGnQZ4TacUfXF2uWR_I5Q0a2ncVw")

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()