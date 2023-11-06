# Offensive Language Identification -  Ranking words by offensiveness

This repository contains our code for our approach in offensive
language identification. Our experiments are based on the OLID
dataset that was made public in 2019. In addition to tackling the
offensive language identification task, we also attempt to
gain a insight into which words the models consider to be
the most effective and how much these words alone affect
the model's decision making.
---

## Setup

First place the datset into the _data_ directory.
Then install the required packages using:

``
pip install -r requirements.txt
``
---
## SVC experiments

All of the functionality regarding the SVM is contained in the OLID_svc.py
script. It trains an SVC on the provided dataset using the command line arguments to set the hyperparameters. Can be set
to finetune mode to automatically find the best parameters.
````
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
````
An auxiliary python script _svc\_helper.py_ is also provided that can be used to
reproduce our experiments. More details can be found in the script itself.

- To reproduce the grid search for the baseline SVC model:

```python OLID_svc.py -t data/train.tsv -d data/dev.tsv --finetune --seed 1```

- To train the baseline model with the best hyperparameters we found:

``python OLID_svc.py -t data/train.tsv -d data/dev.tsv --seed 1 
--penalty l1 --loss squared_hinge --C 0.5 --balanced_weight``

- To reproduce the grid search for the SVC with the enhanced feature set:

``python OLID_svc.py -t data/train.tsv -d data/dev.tsv --seed 1 --finetune --trigram
--stem --tfidf``

- To train the SVC with the enhanced feature set with the best hyperparemeters
we found:

``python OLID_svc.py -t data/train.tsv -d data/dev.tsv --seed 1 --penalty l1 
--loss squared_hinge --C 0.5 --tfidf --stem --trigram``

- To run the experiment for the most offensive words perceived
by the model

``python OLID_svc.py -t data/train.tsv -d data/dev.tsv --test_file data/test.tsv
--seed 1 --penalty l1  --loss squared_hinge --C 0.5 --tfidf 
--stem --trigram --offensiveness``

## BERT experiments

In order to run the BERT experiments, please find the BERT.ipynb notebook inside the src folder. 
All code blocks can be run sequentially in order to perform the grid search (be aware that takes a long time) and to get the most frequent word list.
