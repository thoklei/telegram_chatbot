import numpy as np
import tensorflow as tf 
import re 
import os
import sys
import pickle

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


def pickle_rick(checkpoint_dir, file, name):
    ### pickle-Riiiiick! ###
    output = open(os.path.join(checkpoint_dir, name+'.pkl'), 'wb')
    pickle.dump(file, output)
    output.close()

def unpickle(path, name):
    pkl_file = open(os.path.join(path, name+'.pkl'), 'rb')
    lazarus = pickle.load(pkl_file)
    pkl_file.close()
    return lazarus

def split_input_target(chunk):
    """
    Creates input/target pairs for text generation: sequence up to second-to-last element and
    sequence from second element to last.
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(config):
    """
    Creates a keras sequential model according to specifications.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(config.vocab_size, config.embedding_dim,
                                    batch_input_shape=[config.batch_size, None]),
        tf.keras.layers.LSTM(config.rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(config.vocab_size)
    ])

    return model


def train_model(checkpoint_dir, text_as_int, model, config):
    """
    Given a model, data, and some parameters, this function trains the model on the dataset and returns the trained model.
    """
        
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(config.seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE, drop_remainder=True)

    # def loss(labels, logits):
    #     return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    ### setting callbacks ###

    # The idea here is to store a release version of the model at the end of each episode,
    # so that I can safely abort the training process at any point in time and always
    # keep the most recent version of the exported model.
    class ReleaseCallback(tf.keras.callbacks.Callback):

        def __init__(self):
            super(ReleaseCallback, self).__init__()

            self.export_model = build_model(config.get_release_config())

        def on_train_batch_end(self, batch, logs=None):
            self.export_model.set_weights(self.model.get_weights())
            self.export_model.save(os.path.join(checkpoint_dir, "checkpoint_release.h5"))

    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_dir,"logs")),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir,"checkpoint.h5"),
                                            #save_best_only=True,
                                            period=1),
        ReleaseCallback()
    ]

    ### training - you gotta love keras ###
    model.fit(dataset, epochs=config.EPOCHS, callbacks=callbacks)


class Config():

    ### hyperparameters ###
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    EPOCHS = 15

    embedding_dim = 128
    rnn_units = 1024

    seq_length = 30

    def __init__(self, vocab_size, epochs):
        self.vocab_size = vocab_size
        self.EPOCHS = epochs

    def get_release_config(self):
        c = Config(self.vocab_size, self.epochs)
        c.BUFFER_SIZE = self.BUFFER_SIZE
        c.BATCH_SIZE = 1
        c.embedding_dim = self.embedding_dim
        c.rnn_units = self.rnn_units
        c.seq_length = self.seq_length
        return c 
            