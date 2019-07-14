import tensorflow as tf
import numpy as np
import argparse
from util import prepare_text, remove_unknowns, split_input_target, build_model, Config, train_model, pickle_rick, unpickle

# in case you want to use the Shakespeare dataset to check if it works
#path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

def preprocessing(text, checkpoint_dir, minocc):
    """
    This time, we cannot leave the file as it is. We have to modify it first.
    - replace "\n" by " \n " -> newline is a word
    - insert space between punctuation and last word of sentence
    - create vocab, but only for those words that occur more than once
    - replace all words that occur too seldomly with "<unk>"

    returns the list of integers we will use as the dataset as well as char2idx and idx2char
    """
     
    splitted = prepare_text(text)

    print("Total number of words:",len(splitted))

    occurences = dict()
    for word in splitted:
        if word in list(occurences.keys()):
            occurences[word] += 1
        else:
            occurences[word] = 1
            
    vocab = ["<unk>"]
    for word in list(occurences.keys()):
        if occurences[word] > minocc:
            vocab.append(word)

    splitted = remove_unknowns(vocab, splitted) # removing words that appear less than two times


    print(splitted[0:250])

    print("Number of unique relevant words:", len(vocab))

    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    pickle_rick(checkpoint_dir, char2idx, 'char2idx')
    pickle_rick(checkpoint_dir, idx2char, 'idx2char')
    pickle_rick(checkpoint_dir, splitted, 'dataset')

    return splitted, char2idx, idx2char


def main(training_from_scratch, filename, checkpoint_dir, checkpoint, epochs, minocc):

    if( training_from_scratch ):

        text = open(filename, 'rb').read().decode(encoding='utf-8')
        text, char2idx, idx2char = preprocessing(text, checkpoint_dir, minocc) # note that we are replacing the text here

        vocab_size = len(idx2char)
        config = Config(vocab_size, epochs)

        model = build_model(config)
    else:

        model = tf.keras.models.load_model(checkpoint)
        char2idx = unpickle(checkpoint_dir,'char2idx')
        idx2char = unpickle(checkpoint_dir, 'idx2char')
        text = unpickle(checkpoint_dir, 'dataset')

        vocab_size = len(idx2char)
        config = Config(vocab_size, epochs)

    text_as_int = np.array([char2idx[c] for c in text]) # works because text is a list of words
    train_model(checkpoint_dir, text_as_int, model, config)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains a word-level LSTM, either from scratch or existing checkpoint.')
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
    parser.add_argument("-minocc", type=int, default=1,
                    help="How often a word needs to appear before it shows up in the vocab")

    args = parser.parse_args()
    if args.scratch == 0:
        scratch = False
    else:
        scratch = True
    main(scratch, args.textfile, args.checkpointdir, args.checkpoint, args.epochs, args.minocc)
