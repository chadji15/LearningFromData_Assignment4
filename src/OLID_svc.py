import argparse
from pprint import pprint

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import pandas as pd


from utils import read_corpus, preprocess, binarize_labels
import numpy

SEED = 1
numpy.random.seed(seed=SEED)


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
        "--bigram", action="store_true", help="Add bigram features to the vectorizer"
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
        "--feature_experiments",
        action="store_true",
        help="This puts the program in experiment mode, where various feature settings will be explored"
             "for the given algorithm and reported in the end.",
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
        "--stem",
        action='store_true',
        help="Use stemming"
    )
    parser.add_argument(
        '--bigram',
        action='store_true',
        help='Use bigrams as additional features'
    )
    parser.add_argument(
        '--tfidf',
        action='store_true',
        help='Use TFIDF vectorizer instead of bag of words.'
    )
    args_arr = parser.parse_args()
    return args_arr


def finetune_LinearSVC(args, X_train, Y_train, X_val, Y_val):
    """
    Grid search for LinearSVC over penalty, regularization parameter and loss function
    Parameters
    ----------
    args
    vec: The vectorizer
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
            "cls__C": [ 0.1, 0.5, 1, 2, 4],
            "cls__loss": ["hinge", "squared_hinge"],
        },
        {"cls__penalty": ["l1"], "cls__C": [ 0.1, 0.5, 1, 2, 4], "cls__loss": ["squared_hinge"]},
        {
            "cls__penalty": ["l2"],
            "cls__loss": ["hinge", "squared_hinge"],
            "cls__class_weight": ['balanced']
        },
        {"cls__penalty": ["l1"], "cls__loss": ["squared_hinge"], "cls__class_weight": ['balanced']},
    ]

    ind = []
    for i in range(len(X_train) - 1):
        ind.append(-1)

    for i in range(len(X_val) - 1):
        ind.append(0)

    ps = PredefinedSplit(test_fold=ind)

    print("Running Grid Search...")
    cs = GridSearchCV(
        classifier, param_grid=param_grid, n_jobs=-1, scoring="f1", cv=ps, verbose=True
    )
    cs.fit(X_train + X_val, Y_train + Y_val)
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
    pprint(cs.best_params_)
    pprint(res)

    if args.save_model_path:
        joblib.dump(classifier, args.save_model_path)
        print(f"Model saved to {args.save_model}")


def identity(x):
    return x


def make_pipeline(args):
    # bag of words
    ngram = (1, 1)
    if args.bigram:
        ngram = (1, 2)
    if args.tfidf:
        vec = TfidfVectorizer(
            preprocessor=identity,
            tokenizer=identity,
            token_pattern=None,
            ngram_range=ngram,
        )
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
    model = LinearSVC(penalty=args.penalty, loss=args.loss, C=args.C, class_weight=class_weight, random_state=args.seed,
                      dual='auto', max_iter=10000)

    classifier = Pipeline([("vec", vec), ("cls", model)])
    return classifier


def evaluate(cls, X_val, Y_val):
    print("Evaluating classifier...")
    Y_pred = cls.predict(X_val)
    precision, recall, fscore, support = precision_recall_fscore_support(Y_val, Y_pred, average='binary')
    cm = confusion_matrix(Y_val, Y_pred)
    accuracy = accuracy_score(Y_val, Y_pred,)
    return {
        "cm": cm,
        "precision": precision,
        "recall": recall,
        "f-score": fscore,
        "accuracy": accuracy
    }


def run(args, X_train, Y_train, X_val, Y_val):
    classifier = make_pipeline(args)

    # Train the whole pipline on the training data
    print("Training classifier...")
    classifier.fit(X_train, Y_train)
    # evaluate
    res = evaluate(classifier, X_val, Y_val)
    print("Results: \n-------------------------\n")
    pprint(args)
    pprint(res)

    if args.save_model_path:
        joblib.dump(classifier, args.save_model_path)
        print(f"Model saved to {args.save_model}")


def main():
    args = create_arg_parser()
    numpy.random.seed(args.seed)
    if args.penalty == 'l1' and args.loss == 'hinge':
        raise Exception("The combination of penalty='l1' and loss='hinge' is not supported.")
    print("Reading dataset...")
    # read dataset
    X_train, y_train = read_corpus(args.train_file)
    X_val, y_val = read_corpus(args.dev_file)

    print("Preprocessing dataset...")
    # preprocess and tokenize tweets
    if not args.stem:
        args.stem = False
    X_train = preprocess(X_train, args.stem)
    X_val = preprocess(X_val, args.stem)

    #binarize labels
    y_train_bin = binarize_labels(y_train)
    y_val_bin = binarize_labels(y_val)

    if args.finetune:
        finetune_LinearSVC(args, X_train, y_train_bin, X_val, y_val_bin)
    else:
        run(args, X_train, y_train_bin, X_val, y_val_bin)


if __name__ == '__main__':
    main()
