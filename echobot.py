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
import argparse
import pickle
import os
import tensorflow as tf 
from telegram.error import NetworkError, Unauthorized
from time import sleep
from language_model import WordLanguageModel, CharLanguageModel

update_id = None
lang_model = None


def main(word_level, path_to_model):
    """Run the bot."""
    global update_id
    global lang_model

    model = tf.keras.models.load_model(os.path.join(path_to_model,"checkpoint_release.h5"))

    pkl_file = open(os.path.join(path_to_model,'char2idx.pkl'), 'rb')
    char2idx = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(os.path.join(path_to_model,'idx2char.pkl'), 'rb')
    idx2char = pickle.load(pkl_file)
    pkl_file.close()

    if( word_level ):
        lang_model = WordLanguageModel(model, char2idx, idx2char)        
    else:
        lang_model = CharLanguageModel(model, char2idx, idx2char)

    # Telegram Bot Authorization Token
    bot = telegram.Bot(open("access_token.txt", 'r').read())

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
            global lang_model
            res = lang_model.answer(update.message.text)
            update.message.reply_text(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a word-level LSTM, either from scratch or existing checkpoint.')
    parser.add_argument('-word', type=int,
                    help='1 = use word level text gen, 0 = use char level text gen')
    parser.add_argument('-path', type=str,
                    help="Path to model directory (should contain checkpoint file and vocabulary)")
    args = parser.parse_args()
    if args.word == 1:
        word = True
    else:
        word = False
    main(word, args.path)