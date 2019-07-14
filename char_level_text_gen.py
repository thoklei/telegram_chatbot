import tensorflow as tf
import numpy as np
import argparse
from util import build_model, Config, train_model, pickle_rick

# in case you want to use the Shakespear dataset to check if it works
#path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


def create_vocab_from_file(text, checkpoint_dir):
    """
    Creates a vocabulary from a raw string.

    Returns two dictionaries, char2idx and idx2char, which allow you to translate a character to an index or an index back to a character. 
    """
    
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    pickle_rick(checkpoint_dir, char2idx, 'char2idx')
    pickle_rick(checkpoint_dir, idx2char, 'idx2char')

    return char2idx, idx2char


def main(training_from_scratch, args):

    ### opening the file ###
    text = open(args.filename, 'rb').read().decode(encoding='utf-8')
    vocab_size = len(set(text))

    config = Config(vocab_size, args.epochs, args.initepochs)

    ### looking at shit ###
    print('Length of text: {} characters'.format(len(text)))
    print(text[:250])

    ### creating vocab, converting text to long integer sequence ###
    char2idx, idx2char = create_vocab_from_file(text, args.checkpoint_dir)
    text_as_int = np.array([char2idx[c] for c in text])

    if( training_from_scratch ):
        model = build_model(config)

    else:
        model = tf.keras.models.load_model(args.checkpoint)

    train_model(args.checkpoint_dir, text_as_int, model, config)



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
    parser.add_argument("-initepochs", type=int, default=0,
                    help="The number of the first epoch while training")

    args = parser.parse_args()
    if args.scratch == 0:
        scratch = False
    else:
        scratch = True
    main(scratch, args)
