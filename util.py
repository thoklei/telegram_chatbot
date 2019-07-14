import numpy as np
import tensorflow as tf 
import re 
import os
import sys

def prepare_text(text):
    text = text.replace("\n", " \n ") # making sure that we get paragraphs right
    text = re.sub(r"([?.!,¿])", r" \1 ", text) # inserting space in between punctuation
    splitted = text.split(" ")
    return splitted

def remove_unknowns(vocab, sentence):
    """
    assuming sentence is a list
    """
    for i in range(len(sentence)):
        if not sentence[i] in vocab:
            sentence[i] = "<unk>"
    return sentence