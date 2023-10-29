from utils import read_corpus
import matplotlib.pyplot as plt

X_train, y_train = read_corpus('data\\train.tsv')
plt.hist(y_train)
plt.show()
