
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import time
import argparse

# in case you want to use the Shakespear dataset to check if it works
#path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


def create_vocab_from_file(text):
    """
    Creates a vocabulary from a raw string.

    Returns two dictionaries, char2idx and idx2char, which allow you to translate a character to an index or an index back to a character. 
    """
    
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    return char2idx, idx2char


def split_input_target(chunk):
    """
    Creates input/target pairs for text generation: sequence up to second-to-last element and
    sequence from second element to last.
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Creates a keras sequential model according to specifications.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model


class Config():

    ### hyperparameters ###
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    EPOCHS = 15

    embedding_dim = 256
    rnn_units = 1024

    seq_length = 75

    def __init__(self, vocab_size, epochs):
        self.vocab_size = vocab_size
        self.EPOCHS = epochs
            

def train_model(checkpoint_dir, text_as_int, model, config):
    """
    Given a model, data, and some parameters, this function trains the model on the dataset and returns the trained model.
    """
        
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(config.seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE, drop_remainder=True)

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    ### setting callbacks ###

    # The idea here is to store a release version of the model at the end of each episode,
    # so that I can safely abort the training process at any point in time and always
    # keep the most recent version of the exported model.
    class ReleaseCallback(tf.keras.callbacks.Callback):

        def __init__(self):
            super(ReleaseCallback, self).__init__()

            self.export_model = build_model(
                    vocab_size    = config.vocab_size,
                    embedding_dim = config.embedding_dim,
                    rnn_units     = config.rnn_units,
                    batch_size    = 1)

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

    # I bet there is a more organic way of doing this, but:
    # The stored model has a batchsize of 64, but to run predictions, we need a batchsize of 1.
    # So I rebuild the same architecture with a batchsize of 1, set its weights to the trained weights,
    # store that new model and return the prediction-version of it. 

    new_model = build_model(
                    vocab_size    = config.vocab_size,
                    embedding_dim = config.embedding_dim,
                    rnn_units     = config.rnn_units,
                    batch_size    = 1)
    new_model.set_weights(model.get_weights())
    #new_model.save(os.path.join(checkpoint_dir, "checkpoint_release.h5"))

    return new_model


def generate_text(model, start_string, char2idx, idx2char):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 100

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (''.join(text_generated))


def main(training_from_scratch, filename, checkpoint_dir, checkpoint, epochs):

    print("from scratch: ",training_from_scratch)

    ### opening the file ###
    text = open(filename, 'rb').read().decode(encoding='utf-8')
    vocab_size = len(set(text))

    config = Config(vocab_size, epochs)

    ### looking at shit ###
    print('Length of text: {} characters'.format(len(text)))
    print(text[:250])

    ### creating vocab, converting text to long integer sequence ###
    char2idx, idx2char = create_vocab_from_file(text)
    text_as_int = np.array([char2idx[c] for c in text])

    if( training_from_scratch ):
        model = build_model(
                    vocab_size    = config.vocab_size,
                    embedding_dim = config.embedding_dim,
                    rnn_units     = config.rnn_units,
                    batch_size    = config.BATCH_SIZE)

    else:
        model = tf.keras.models.load_model(checkpoint)

    model = train_model(checkpoint_dir, text_as_int, model, config)

    res = generate_text(model, start_string=u"Dafür dass dir wollen und können", char2idx=char2idx, idx2char=idx2char)
    
    print(res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains a character-level LSTM, either from scratch or existing checkpoint.')
    parser.add_argument('-scratch', type=int,
                    help='0 = starting from checkpoint, 1 = staring from scratch')
    parser.add_argument('-textfile', type=str,
                    help='Which file to use as training data')
    parser.add_argument("-checkpointdir", type=str,
                    help="path to checkpoint directory from which to start")
    parser.add_argument("-checkpoint", type=str,
                    help="path to checkpoint file from which to start")
    parser.add_argument("-epochs", type=int,
                    help="how many epochs do you want to run?")

    args = parser.parse_args()
    if args.scratch == 0:
        scratch = False
    else:
        scratch = True
    main(scratch, args.textfile, args.checkpointdir, args.checkpoint, args.epochs)
