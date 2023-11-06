"""
Trains an SVC on the provided dataset using the command line arguments to set the hyperparameters. Can be set
to finetune mode to automatically find the best parameters.

usage: OLID_svc.py [-h] [-tf TRAIN_FILE] [-df DEV_FILE]
                   [--test_file TEST_FILE] [--trigram] [--stem] [--finetune]
                   [--C C] [--loss {hinge,squared_hinge}] [--penalty {l1,l2}]
                   [--balanced_weight] [--save_model_path SAVE_MODEL_PATH]
                   [--seed SEED] [--tfidf] [--remove_emojis]
                   [--remove_handles] [--offensiveness]

options:
  -h, --help            show this help message and exit
  -tf TRAIN_FILE, --train_file TRAIN_FILE
                        Train file to learn from (default train.tsv)
  -df DEV_FILE, --dev_file DEV_FILE
                        Dev file to evaluate on (default dev.tsv)
  --test_file TEST_FILE
                        Path to the test set file. Optional
  --trigram             Add bigram and trigram features to the vectorizer
  --stem                Use stemming on the corpus
  --finetune            Call the finetune method for the selected algorithm
  --C C                 Regularization parameter for LSVC
  --loss {hinge,squared_hinge}
                        Loss function for lsvc
  --penalty {l1,l2}     Penaly type for LinearSVC
  --balanced_weight     Use the values of y to automatically adjust weights in
                        SVC inversely proportional to class frequencies in the
                        input data. Overrides C.
  --save_model_path SAVE_MODEL_PATH
                        Save the model to the provided path
  --seed SEED           Random seed initialization
  --tfidf               Use TFIDF vectorizer instead of bag of words.
  --remove_emojis       Replace emojis with the token EMJ
  --remove_handles      Completely eliminate @USER handles from the documents
  --offensiveness       Additionally test for offensiveness of words and build
                        a filter based on it. Is overriden by finetune.

"""

import argparse
from pprint import pprint
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import pandas as pd
from utils import read_corpus, preprocess, binarize_labels
import numpy as np
import random
from collections import defaultdict


def create_arg_parser():
    """
    Define the command line arguments. Description for each argument in the help strings.
    Takes no parameters.

    Returns
    -------
    args (object): an object containing all the command line arguments, accessible as properties of the object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tf",
        "--train_file",
        default="train.tsv",
        type=str,
        help="Train file to learn from (default train.tsv)",
    )
    parser.add_argument(
        "-df",
        "--dev_file",
        default="dev.tsv",
        type=str,
        help="Dev file to evaluate on (default dev.tsv)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to the test set file. Optional"
    )
    parser.add_argument(
        "--trigram", action="store_true", help="Add bigram and trigram features to the vectorizer"
    )
    parser.add_argument(
        "--stem", action="store_true", help="Use stemming on the corpus"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Call the finetune method for the selected algorithm",
    )
    parser.add_argument(
        "--C", type=float, default=0.5, help="Regularization parameter for LSVC"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["hinge", "squared_hinge"],
        default="squared_hinge",
        help="Loss function for lsvc",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Penaly type for LinearSVC",
    )
    parser.add_argument(
        "--balanced_weight",
        action="store_true",
        help="Use the values of y to automatically adjust weights in SVC inversely proportional to class "
             "frequencies in the input data. Overrides C."
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        help="Save the model to the provided path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed initialization"
    )
    parser.add_argument(
        '--tfidf',
        action='store_true',
        help='Use TFIDF vectorizer instead of bag of words.'
    )
    parser.add_argument(
        '--remove_emojis',
        action='store_true',
        help='Replace emojis with the token EMJ'
    )
    parser.add_argument(
        '--remove_handles',
        action='store_true',
        help='Completely eliminate @USER handles from the documents'
    )
    parser.add_argument(
        '--offensiveness',
        action='store_true',
        help='Additionally test for offensiveness of words and build  a filter based on it. Is overriden by finetune.'
    )
    args_arr = parser.parse_args()
    return args_arr


def finetune_LinearSVC(args, X_train, Y_train, X_val, Y_val):
    """
    Grid search for LinearSVC over penalty, regularization parameter and loss function
    Parameters
    ----------
    args
    X_train
    Y_train
    X_val
    Y_val

    Returns
    -------
    None
    """
    classifier = make_pipeline(args)

    # Two grids because l1 penalty does not work with hinge loss
    # times two because class weight overrides C
    param_grid = [
        {
            "cls__penalty": ["l2"],
            "cls__C": [0.1, 0.5, 1, 2, 4],
            "cls__loss": ["hinge", "squared_hinge"],

        },
        {"cls__penalty": ["l1"],
         "cls__C": [0.1, 0.5, 1, 2, 4],
         "cls__loss": ["squared_hinge"],
         "cls__class_weight": ['balanced', None]
         },
    ]

    # Create a pre-defined split so that the validation set is always used for validation
    ind = []
    for i in range(len(X_train) - 1):
        ind.append(-1)

    for i in range(len(X_val) - 1):
        ind.append(0)

    ps = PredefinedSplit(test_fold=ind)

    print("Running Grid Search...")
    cs = GridSearchCV(
        classifier, param_grid=param_grid, n_jobs=-1, scoring="f1_macro", cv=ps, verbose=True
    )
    cs.fit(X_train + X_val, Y_train + Y_val)
    # Extract data frame with parameters for each run and performance on validation set
    df = pd.concat(
        [
            pd.DataFrame(cs.cv_results_["params"]),
            pd.DataFrame(cs.cv_results_["mean_test_score"], columns=["f1-score"]),
        ],
        axis=1,
    )
    print("Results: \n-------------------------\n")
    print(df)
    best_model = cs.best_estimator_
    res = evaluate(best_model, X_val, Y_val)
    print("Evaluation of best model on dev set: \n-------------------------\n")
    pprint(cs.best_params_)
    pprint(res)

    # Save the model using joblib
    if args.save_model_path:
        joblib.dump(best_model, args.save_model_path)
        print(f"Model saved to {args.save_model}")


def identity(x):
    return x


def make_pipeline(args):
    """
    This function creates the sklearn pipeline object based on the command line arguments.

    Parameters
    ----------
    args

    Returns
    -------
    Pipeline object with vectorizer and classifier
    """
    # uni gram
    ngram = (1, 1)
    if args.trigram:
        ngram = (1, 3)
    # Use tfidf vectorizer
    # Dont do preprocessing or tokenization because we do it beforehand
    if args.tfidf:
        vec = TfidfVectorizer(
            preprocessor=identity,
            tokenizer=identity,
            token_pattern=None,
            ngram_range=ngram,
        )
    # Else use bag-of-words
    else:
        vec = CountVectorizer(
            preprocessor=identity,  # we already preprocessed the tweets
            tokenizer=identity,  # we already tokenized them
            token_pattern=None,
            ngram_range=ngram
        )

    class_weight = None
    if args.balanced_weight:
        class_weight = 'balanced'
    # Create the model and set seed
    model = LinearSVC(penalty=args.penalty, loss=args.loss, C=args.C, class_weight=class_weight, random_state=args.seed,
                      dual='auto', max_iter=10000)

    # Combine both into the pipeline and return
    classifier = Pipeline([("vec", vec), ("cls", model)])
    return classifier


def evaluate(cls, X_val, Y_val):
    """
    Calculates performance metrics of classifier cls based on the dataset provided

    Parameters
    ----------
    cls
    X_val
    Y_val

    Returns
    -------
    Dictionary with confusion matrix, precision, recall, f1-score and accuracy

    """
    print("Evaluating classifier...")
    Y_pred = cls.predict(X_val)
    precision, recall, fscore, support = precision_recall_fscore_support(Y_val, Y_pred, average='macro')
    cm = confusion_matrix(Y_val, Y_pred)
    accuracy = accuracy_score(Y_val, Y_pred, )
    return {
        "cm": cm,
        "precision": precision,
        "recall": recall,
        "f-score": fscore,
        "accuracy": accuracy
    }


def evaluate_test_set(args, classifier):
    """
    Evaluate the classifier on the test set

    Parameters
    ----------
    args
    classifier

    Returns
    -------
    Dictionary with confusion matrix, precision, recall, f1-score and accuracy


    """
    X_test, y_test = read_corpus(args.test_file)
    X_test = preprocess(X_test, args.stem, args.remove_emojis, args.remove_handles)
    y_test_bin = binarize_labels(y_test)
    res = evaluate(classifier, X_test, y_test_bin)
    return res


def run(args, X_train, Y_train, X_val, Y_val):
    """
    Trains and evaluates a single model based on the command line arguments
    Parameters
    ----------
    args
    X_train
    Y_train
    X_val
    Y_val

    Returns
    -------
    None
    """
    classifier = make_pipeline(args)

    # Train the whole pipline on the training data
    print("Training classifier...")
    classifier.fit(X_train, Y_train)
    # evaluate
    res = evaluate(classifier, X_val, Y_val)
    print("Results for dev set: \n-------------------------\n")
    pprint(args)
    pprint(res)

    if args.test_file:
        res = evaluate_test_set(args, classifier)
        print("Results for test set: \n-------------------------\n")
        pprint(res)

    if args.save_model_path:
        joblib.dump(classifier, args.save_model_path)
        print(f"Model saved to {args.save_model}")


def make_dataset(args):
    """
    Reads the dataset based on the command line arguments, applies preprocessing to the documents and binarizes the
    labels

    Parameters
    ----------
    args

    Returns
    -------
    X_train, y_train_bin, X_val, y_val_bin
    """
    print("Reading dataset...")
    # read dataset
    X_train, y_train = read_corpus(args.train_file)
    X_val, y_val = read_corpus(args.dev_file)

    print("Preprocessing dataset...")
    # preprocess and tokenize tweets

    X_train = preprocess(X_train, args.stem, args.remove_emojis, args.remove_handles)
    X_val = preprocess(X_val, args.stem, args.remove_emojis, args.remove_handles)

    # binarize labels
    y_train_bin = binarize_labels(y_train)
    y_val_bin = binarize_labels(y_val)

    return X_train, y_train_bin, X_val, y_val_bin


def compute_word_frequency(X_train):
    """

    Counts how many tweets every word in the corpus appears in.

    Parameters
    ----------
    X_train

    Returns
    -------
    Dictionary with occurences per word
    """
    word_counts = defaultdict(int)
    for sentence in X_train:
        words = sentence.split()
        for word in words:
            word_counts[word] += 1
    return word_counts


def compute_word_frequency_in_offensive_instances(X_train, y_pred):
    """
    For each word in the corpus, count how many tweets that are predicted offensive this word appears in.

    Parameters
    ----------
    X_train
    y_pred

    Returns
    -------
    Dictionary with occurences per word, sorted
    """
    word_counts = defaultdict(int)
    for (sentence, predicted_label) in zip(X_train, y_pred):
        if predicted_label == 1:  # If the data instance is predicted as offensive add it's o
            words = sentence.split()
            for word in words:
                word_counts[word] += 1
    return {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}


def compute_normalized_word_frequency_in_offensive_instances(word_frequency, word_frequency_in_offensive_instances,
                                                             support_threshold):
    """
    Create a dictionary with the metric numberOfOffensiveTweets/numberOfTotalTweets for each word in the corpus.
    Has a support threshold for 10 to discard rare words.

    Parameters
    ----------
    word_frequency
    word_frequency_in_offensive_instances
    support_threshold

    Returns
    -------
    Dictionary
    """
    normalized_word_frequency_in_offensive_instances = defaultdict(float)
    for word in word_frequency_in_offensive_instances:
        if word_frequency[word] != 0 and word_frequency[word] >= support_threshold:
            normalized_word_frequency_in_offensive_instances[word] = word_frequency_in_offensive_instances[word] / \
                                                                     word_frequency[word]

    def threshold(pair):
        k, v = pair
        if v <= 0.5:
            return False
        return True

    filtered_dict = dict(filter(threshold, normalized_word_frequency_in_offensive_instances.items()))
    return {k: v for k, v in
            sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True)}


def compute_offensiveness_metric(X, y_pred):
    """
    For each word in the corpus calculate how offensive our model perceives this word to be

    Parameters
    ----------
    X
    y_pred

    Returns
    -------
    Dictionary

    """
    word_frequency = compute_word_frequency(X)
    word_frequency_in_offensive_instances = compute_word_frequency_in_offensive_instances(X, y_pred)
    normalized_word_frequency_in_offensive_instances = compute_normalized_word_frequency_in_offensive_instances(
        word_frequency, word_frequency_in_offensive_instances, 10)

    return normalized_word_frequency_in_offensive_instances


def classify_based_on_word_list(X, word_list):
    """
    Trivial prediction model. If document contains a word that is part of word_list then predict positive, else
    negative.

    Parameters
    ----------
    X
    word_list

    Returns
    -------
    y_pred: list of ints

    """
    y_pred = []
    for sentence in X:
        prediction = 0
        for word_token in sentence:
            if word_token in word_list:
                prediction = 1
        y_pred.append(prediction)
    return y_pred


def offensive_words(args, X_train, Y_train, X_val, Y_val):
    """
    This is the function that tends to our research question. Trains an SVM based on the command line arguments
    and prints its performance. Then calculate this "offensiveness" metric for each word and construct two simple
    models, one based on the word list obtained from the predictions of the model and one from the ground truth.

    Parameters
    ----------
    args
    X_train
    Y_train
    X_val
    Y_val

    Returns
    -------
    None

    """

    if args.test_file is None:
        raise Exception("Please provide test file")
    classifier = make_pipeline(args)
    # Train the whole pipline on the training data
    print("Training classifier...")
    classifier.fit(X_train, Y_train)
    # evaluate
    res = evaluate(classifier, X_val, Y_val)
    print("Results for dev set: \n-------------------------\n")
    pprint(args)
    pprint(res)

    if args.test_file:
        res = evaluate_test_set(args, classifier)
        print("Results for test set: \n-------------------------\n")
        pprint(res)

    # Combine training and validation set
    X = X_train + X_val
    Y = Y_train + Y_val
    # Get the prediction on the training set
    y_pred_train = classifier.predict(X)

    # Read the test set
    X_test, y_test = read_corpus(args.test_file)
    X_test = preprocess(X_test, args.stem, args.remove_emojis, args.remove_handles)
    y_test_bin = binarize_labels(y_test)

    # Offensiveness metric for each word based on the predictions
    normalized_word_frequency_in_offensive_instances_train = compute_offensiveness_metric(X, y_pred_train)
    # Get the 100 most offensive words and make predictions using this list
    word_list_train = list(normalized_word_frequency_in_offensive_instances_train.keys())
    y_pred_word_list = classify_based_on_word_list(X_test, word_list_train)
    # Calculate f1 score of the simple model and agreement percentage with the SVM
    list_model_f1 = f1_score(y_test_bin, y_pred_word_list, average='macro')
    agreement_percent = accuracy_score(y_test_bin, y_pred_word_list)

    # Do the same but this time using the ground truth labels
    offensiveness_metric_raw_data = compute_offensiveness_metric(X, Y)
    word_list_raw_data = list(offensiveness_metric_raw_data.keys())
    y_pred_word_list_raw_data = classify_based_on_word_list(X_test, word_list_raw_data)
    trivial_model_f1 = f1_score(y_test_bin, y_pred_word_list_raw_data, average='macro')

    print(f'''---------------
Most offensive words based on predictions
----------------    
''')
    pprint(word_list_train)
    print(f'''
F1-Score of filter based on this list: {list_model_f1}
Agreement with SVC percent: {agreement_percent}
--------------

Most offensive words based on ground truth
--------------
''')
    pprint(word_list_raw_data)
    print(f'''
F1-Score of filter based on this metric: {trivial_model_f1}
''')


def main():
    args = create_arg_parser()
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.penalty == 'l1' and args.loss == 'hinge':
        raise Exception("The combination of penalty='l1' and loss='hinge' is not supported.")
    if args.finetune and args.offensiveness:
        raise Exception("The combination of finetune and offensiveness flags is not supported.")

    X_train, y_train_bin, X_val, y_val_bin = make_dataset(args)

    if args.finetune:
        finetune_LinearSVC(args, X_train, y_train_bin, X_val, y_val_bin)
    elif args.offensiveness:
        offensive_words(args, X_train, y_train_bin, X_val, y_val_bin)
    else:
        run(args, X_train, y_train_bin, X_val, y_val_bin)


if __name__ == '__main__':
    main()
