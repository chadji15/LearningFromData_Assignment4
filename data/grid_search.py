import random as python_random
import argparse
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay, CosineDecay
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline
from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[:-1]).strip())
            labels.append(tokens.split()[-1])
    return documents, labels


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='data/train.tsv', type=str,
                        help="Input file to learn from (default train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default='data/dev.tsv',
                        help="Separate dev set to read in (default dev.tsv)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    args = parser.parse_args()
    return args


def train_model(lm, tokens_train, Y_train_bin, num_labels, epochs, batch_size, learning_rate):
    print("Loading model....")
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=num_labels)
    loss_function = BinaryCrossentropy(from_logits=True)
    num_decay_steps = len(Y_train_bin) * epochs
    if learning_rate == "PolynomialDecay":
        lr_scheduler = PolynomialDecay(
            initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_decay_steps
        )
    elif learning_rate == "CosineDecay":
        lr_scheduler = CosineDecay(
            initial_learning_rate=5e-5, decay_steps = num_decay_steps
        )
    else:
        lr_scheduler = learning_rate
    optim = Adam(learning_rate=lr_scheduler)
    print("Training model....")
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epochs,
              batch_size=batch_size)
    print("Done!")
    return model


def evaluate_model(lm, tokens_dev, Y_dev_bin, labels):
    print("Evaluating model....")
    pred = lm.predict(tokens_dev)["logits"]
    # Get predictions using the trained model
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_dev_bin, axis=1)

    report = classification_report(Y_test, Y_pred, target_names=labels, digits=3)
    print(report)
#    cm = confusion_matrix(Y_test, Y_pred)
#    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#    disp.plot()
#    plt.savefig(figpath)
    return accuracy_score(Y_test, Y_pred), f1_score(Y_test, Y_pred)

def create_param_grid():
    param_grid = {'epochs': [1, 2, 3, 4, 5], 'max_seq_len': [50, 100, 150], 
                  'batches': [16, 32, 64], 
                  'lr_schedulers': ["PolynomialDecay", "CosineDecay", 5e-5, 4e-5, 3e-5]}
    keys, values = zip(*param_grid.items())
    result = [dict(zip(keys, p)) for p in product(*values)]
    return result

def main():
    lm = 'bert-base-uncased'

    args = create_arg_parser()
    # Read in the data and embeddings
    print("..........................\n")
    print("Loading data...")
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    X_test, Y_test = read_corpus(args.test_file)

    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    Y_test_bin = encoder.fit_transform(Y_test)
    labels = encoder.classes_
    Y_train_bin = np.hstack((1 - Y_train_bin, Y_train_bin))
    Y_dev_bin = np.hstack((1 - Y_dev_bin, Y_dev_bin))
    Y_test_bin = np.hstack((1 - Y_test_bin, Y_test_bin))
    
    param_grid = create_param_grid()

    performances = []
    models = []
    max_accuracy = 0
    i = 0
    seeds = [1234, 123, 12, 21, 321]
    for seed in seeds:
        for parameters in param_grid:
            tf.keras.backend.clear_session()
            np.random.seed(seed)
            tf.random.set_seed(seed)
            python_random.seed(seed)
            tokenizer = AutoTokenizer.from_pretrained(lm)
            tokens_train = tokenizer(X_train, padding=True, max_length=parameters['max_seq_len'],
                                     truncation=True, return_tensors="np").data
            tokens_dev = tokenizer(X_dev, padding=True, max_length=parameters['max_seq_len'],
                                   truncation=True, return_tensors="np").data
            tokens_test = tokenizer(X_test, padding=True, max_length=parameters['max_seq_len'],
                                   truncation=True, return_tensors="np").data
            model = train_model(lm, tokens_train, Y_train_bin,  len(labels),
                                epochs=parameters['epochs'], batch_size=parameters['batches'], learning_rate=parameters['lr_schedulers'])

            print(parameters)
            acc, f1 = evaluate_model(model, tokens_dev, Y_dev_bin, labels)
            performances.append(acc)
            if i == 0:
                parameters['accuracy'] = [acc]
                parameters['f1'] = [f1]
            else:
                parameters['accuracy'].append(acc)
                parameters['f1'].append(f1)
        i += 1
    output_file = open('results.txt', 'w', encoding='utf-8')
    for dic in param_grid:
      json.dump(dic, output_file) 
      output_file.write("\n")
    

if __name__ == "__main__":
    main()