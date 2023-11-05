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

def preprocess(documents, stem=False):
    # remove emojis
    docs = []
    tk = TweetTokenizer(
        preserve_case=False, #convert to lowercase
        reduce_len=True, # replace repeated character sequences of length 3 or greater with sequences of length 3
        strip_handles=True, #remove twitter handles
    )
    ps = PorterStemmer()
    for doc in documents:
        s = demoji.replace(doc, "EMJ")
        toks = tk.tokenize(s)
        if stem:
            toks = [ps.stem(word) for word in toks]
        s = ' '.join(toks)
        docs.append(s)
    return docs

def binarize_labels(labels):
    return [1 if s == 'OFF' else 0 for s in labels]