#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adapted from https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/echobot.py

In order to make this work, you will need a Token for your bot which you need to insert in line 45, as well as
a file with the chat data you used to train the model and of course the checkpoint from which to create the language model
(see lines 24 and 25: I don't intend to provide my own chats here or a model that was trained on them.)
"""
import logging
import telegram
import tensorflow as tf 
from telegram.error import NetworkError, Unauthorized
from time import sleep
from char_level_text_gen import generate_text, create_vocab_from_file

update_id = None
model = None
char2idx = None
idx2char = None

def language_model():

    path_to_model = "checkpoint.h5"
    filename = "chats.txt"

    text = open(filename, 'rb').read().decode(encoding='utf-8')

    ### creating vocab, converting text to long integer sequence ###
    char2idx, idx2char = create_vocab_from_file(text)

    model = tf.keras.models.load_model(path_to_model)

    return model, char2idx, idx2char

def main():
    """Run the bot."""
    global update_id
    global model 
    global char2idx
    global idx2char
    model, char2idx, idx2char = language_model()

    # Telegram Bot Authorization Token
    bot = telegram.Bot(open("acces_token.txt", 'r').read())

    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.
    try:
        update_id = bot.get_updates()[0].update_id
    except IndexError:
        update_id = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    while True:
        try:
            echo(bot)
        except NetworkError:
            sleep(1)
        except Unauthorized:
            # The user has removed or blocked the bot.
            update_id += 1


def echo(bot):
    """Echo the message the user sent."""
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=10):
        update_id = update.update_id + 1

        if update.message:  # your bot can receive updates without messages
            global char2idx
            global idx2char
            res = generate_text(model, start_string=update.message.text, char2idx=char2idx, idx2char=idx2char)
            update.message.reply_text(res)


if __name__ == '__main__':
    main()