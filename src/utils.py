"""
This python file contains several utility functions
"""

import demoji
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

def read_corpus(corpus_file):
    """Read in review data set and returns docs and labels"""
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[:-1]).strip())
            labels.append(tokens.split()[-1])
    return documents, labels


def preprocess(documents, stem=False, remove_emojis=False, remove_handles=True):
    """
    Preprocess the documents

    Parameters
    ----------
    documents
    stem
    remove_emojis
    remove_handles

    Returns
    -------
    List of strings
    """

    docs = []
    # Itinitialize tokenizer
    tk = TweetTokenizer(
        preserve_case=False,  # convert to lowercase
        reduce_len=True,  # replace repeated character sequences of length 3 or greater with sequences of length 3
        strip_handles=remove_handles,  # remove twitter handles
    )
    ps = PorterStemmer()
    # For every tweet
    for doc in documents:
        s = doc
        # Replace emojis if specified
        if remove_emojis:
            s = demoji.replace(doc, "EMJ ")
        # Tokenize
        toks = tk.tokenize(s)
        # Stem words if specified
        if stem:
            toks = [ps.stem(word) for word in toks]
        # Join the tokens into a sentence
        s = ' '.join(toks)
        docs.append(s)
    return docs


def binarize_labels(labels):
    return [1 if s == 'OFF' else 0 for s in labels]
