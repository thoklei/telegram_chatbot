import numpy as np 
import re
import abc
import tensorflow as tf 
from util import remove_unknowns, prepare_text

class LanguageModel():

    def __init__(self, model, char2idx, idx2char):
        self.model = model 
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.delimiter = None

    @abc.abstractclassmethod
    def preprocessing(self, sentence):
        pass

    @abc.abstractmethod
    def answer(self, sentence):

        sentence = self.preprocessing(sentence)

        # Number of characters to generate
        num_generate = 100

        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in sentence]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0

        # Here batch size == 1
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

        return (self.delimiter.join(text_generated))



class CharLanguageModel(LanguageModel):

    def __init__(self, model, char2idx, idx2char):
        super(CharLanguageModel, self).__init__(model, char2idx, idx2char)
        self.delimiter = ''

    
    def preprocessing(self, sentence):
        return sentence


class WordLanguageModel(LanguageModel):

    def __init__(self, model, char2idx, idx2char):
        super(WordLanguageModel, self).__init__(model, char2idx, idx2char)
        self.delimiter = " "


    def preprocessing(self, sentence):
        return remove_unknowns(self.idx2char, prepare_text(sentence))

    