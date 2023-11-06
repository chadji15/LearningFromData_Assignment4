"""
This script is meant to automate some of our experiments. It's not part of the main code base, but rather a tool that
is mainly intended to facilitate transparency and reproducability. Not configurable from the command line. For any
changes to hyperparameters and paths the code needs to be changed directly. All of the experiments are run 5 times
and averages and standard deviations are derived where needed.

Modes:
----
Finetune_baseline: Performs a grid search over the parameter space of the SVC using Bag-of-words and simple features.
Finetune_features: Performs a grid search over the parameter space of the SVC using TFIDF, stemming and trigrams.
baseline_stats: Trains an SVC with the hyperparameters specified in the function body 5 times and calculates
    average f1-score and standard deviation
feature_stats: Trains an SVC with TFIDF, stemming and trigrams using
    the hyperparameters specified in the function body 5 times and calculates
    average f1-score and standard deviation
ablation_studies: Determine the contribution of each additional feature
offensiveness_stats: Run 10 experiments for the offensiveness experiment

usage: svc_helper.py [-h]
                 [--mode {finetune_baseline,finetune_features,baseline_stats,feature_stats,ablation_studies}]

options:
  -h, --help            show this help message and exit
  --mode {finetune_baseline,finetune_features,baseline_stats,feature_stats,ablation_studies, offensiveness_stats}

"""

import itertools
import sys

import OLID_svc
from contextlib import redirect_stdout
import random
import numpy as np
import argparse


def get_best_config_baseline():
    with open('results\\finetune_baseline.txt', 'w') as f:
        with redirect_stdout(f):
            args = OLID_svc.create_arg_parser()
            args.finetune = True
            args.train_file = "data//train.tsv"
            args.dev_file = "data//dev.tsv"
            args.remove_emojis = False
            args.remove_handles = False

            for seed in [1, 12, 123, 1234, 12345]:
                np.random.seed(seed)
                random.seed(seed)
                args.seed = seed
                print("Run args:\n")
                print(args)
                X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
                OLID_svc.finetune_LinearSVC(args, X_train, y_train_bin, X_val, y_val_bin)


def get_best_config_features():
    with open('results\\finetune_features.txt', 'w') as f:
        with redirect_stdout(f):
            args = OLID_svc.create_arg_parser()
            args.finetune = True
            args.train_file = "data//train.tsv"
            args.dev_file = "data//dev.tsv"
            args.remove_emojis = False
            args.remove_handles = False
            args.trigram = True
            args.stem = True
            args.tfidf = True
            for seed in [1, 12, 123, 1234, 12345]:
                np.random.seed(seed)
                args.seed = seed
                print("Run args:\n")
                print(args)
                X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
                OLID_svc.finetune_LinearSVC(args, X_train, y_train_bin, X_val, y_val_bin)


def get_baseline_stats():
    with open('results\\baseline_stats.txt', 'w') as f:
        with redirect_stdout(f):
            args = OLID_svc.create_arg_parser()
            args.train_file = "data//train.tsv"
            args.dev_file = "data//dev.tsv"
            args.test_file = "data//test.tsv"
            args.balanced_weight = True
            args.penalty = 'l1'
            args.loss = 'squared_hinge'
            args.C = 0.5
            print(args)
            scores = []
            seeds = [1, 12, 123, 1234, 12345]
            for seed in seeds:
                np.random.seed(seed)
                args.seed = seed
                print(f'''----------
SEED: {seed}
----------''')
                X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
                cls = OLID_svc.make_pipeline(args)
                cls.fit(X_train, y_train_bin)
                res = OLID_svc.evaluate(cls, X_val, y_val_bin)
                scores.append(res['f-score'])

            avg = sum(scores) / len(scores)
            stdd = np.std(scores)
            print(f'''Average F1-Score: {avg}
Standard Deviation: {stdd}''')

            best_seed = seeds[np.argmax(scores)]
            np.random.seed(best_seed)
            args.seed = best_seed
            print(f'''----------
SEED: {best_seed}
----------''')
            X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
            cls = OLID_svc.make_pipeline(args)
            cls.fit(X_train, y_train_bin)
            res = OLID_svc.evaluate_test_set(args, cls)
            print(f"F1 Score on the test set: {res['f-score']}")


def get_features_stats():
    with open('results\\features_stats.txt', 'w') as f:
        with redirect_stdout(f):
            args = OLID_svc.create_arg_parser()
            args.train_file = "data//train.tsv"
            args.dev_file = "data//dev.tsv"
            args.test_file = "data//test.tsv"
            args.balanced_weight = False
            args.penalty = 'l1'
            args.loss = 'squared_hinge'
            args.C = 1
            args.trigram = True
            args.stem = True
            args.tfidf = True
            print(args)
            scores = []
            seeds = [1, 12, 123, 1234, 12345]
            for seed in seeds:
                np.random.seed(seed)
                args.seed = seed
                print(f'''----------
SEED: {seed}
----------''')
                X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
                cls = OLID_svc.make_pipeline(args)
                cls.fit(X_train, y_train_bin)
                res = OLID_svc.evaluate(cls, X_val, y_val_bin)
                scores.append(res['f-score'])

            avg = sum(scores) / len(scores)
            stdd = np.std(scores)
            print(f'''Scores: {scores}
Average F1-Score: {avg}
Standard Deviation: {stdd}''')

            best_seed = seeds[np.argmax(scores)]
            np.random.seed(best_seed)
            args.seed = best_seed
            print(f'''----------
SEED: {best_seed}
----------''')
            X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
            cls = OLID_svc.make_pipeline(args)
            cls.fit(X_train, y_train_bin)
            res = OLID_svc.evaluate_test_set(args, cls)
            print(f"F1 Score on the test set: {res['f-score']}")


def ablation_studies():
    with open('results\\ablation.txt', 'w') as f:
        with redirect_stdout(f):
            for trigram, stem, tfidf in set(itertools.permutations([True, True, False], 3)):
                args = OLID_svc.create_arg_parser()
                args.train_file = "data//train.tsv"
                args.dev_file = "data//dev.tsv"
                args.balanced_weight = False
                args.penalty = 'l1'
                args.loss = 'squared_hinge'
                args.C = 1
                args.trigram = trigram
                args.stem = stem
                args.tfidf = tfidf
                print(args)
                scores = []
                for seed in [1, 12, 123, 1234, 12345]:
                    np.random.seed(seed)
                    args.seed = seed
                    print(f'''----------
SEED: {seed}
----------''')
                    X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
                    cls = OLID_svc.make_pipeline(args)
                    cls.fit(X_train, y_train_bin)
                    res = OLID_svc.evaluate(cls, X_val, y_val_bin)
                    scores.append(res['f-score'])

                avg = sum(scores) / len(scores)
                stdd = np.std(scores)
                print(f'''Scores: {scores}
Average F1-Score: {avg}
Standard Deviation: {stdd}''')


def offensiveness_stats():
    with open('results\\offensiveness_stats .txt', 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            args = OLID_svc.create_arg_parser()
            args.train_file = "data//train.tsv"
            args.dev_file = "data//dev.tsv"
            args.balanced_weight = False
            args.penalty = 'l1'
            args.loss = 'squared_hinge'
            args.C = 1
            args.trigram = True
            args.stem = True
            args.tfidf = True
            args.offensiveness = True
            print(args)
            list_f1_list = []
            agreement_list = []
            seeds = [1, 12, 123, 1234, 12345,2,23,234,2345,23456]
            for seed in seeds:
                np.random.seed(seed)
                args.seed = seed
                print(f'''----------
SEED: {seed}
----------''')
                X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
                list_model_f1, agreement, trivial_f1 = OLID_svc.offensive_words(args,X_train,y_train_bin, X_val, y_val_bin)
                list_f1_list.append(list_model_f1)
                agreement_list.append(agreement)

            print(f'F1-Score of filter based on offensive words from ground truth: {trivial_f1}')
            avg = sum(list_f1_list) / len(list_f1_list)
            stdd = np.std(list_f1_list)
            print(f'''Stats for model based on list from predictions:
Scores: {list_f1_list}
Average F1-Score (on validation set): {avg}
Standard Deviation: {stdd}

Agreement with teacher model:
Scores: {agreement_list}
Average: {sum(agreement_list) / len(agreement_list)}
Standard deviation: {np.std(agreement_list)}''')

            best_seed = seeds[np.argmax(list_f1_list)]
            np.random.seed(best_seed)
            args.seed = best_seed
            args.test_file = 'data//test.tsv'
            print(f'''----------
SEED: {best_seed}
----------''')
            X_train, y_train_bin, X_val, y_val_bin = OLID_svc.make_dataset(args)
            list_model_f1, agreement, trivial_f1 = OLID_svc.offensive_words(args, X_train, y_train_bin, X_val,
                                                                            y_val_bin)
            print(f"F1 Score on the test set: {list_model_f1}. \nAgreement with teacher model: {agreement}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['finetune_baseline', 'finetune_features', 'baseline_stats', 'feature_stats', 'ablation_studies',
                 'offensiveness_stats']
    )
    args1 = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if args1.mode == 'finetune_baseline':
        get_best_config_baseline()
    elif args1.mode == 'finetune_features':
        get_best_config_features()
    elif args1.mode == 'baseline_stats':
        get_baseline_stats()
    elif args1.mode == 'feature_stats':
        get_features_stats()
    elif args1.mode == 'ablation_studies':
        ablation_studies()
    elif args1.mode == 'offensiveness_stats':
        offensiveness_stats()
