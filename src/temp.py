from nltk.tokenize import TweetTokenizer
import demoji

tk = TweetTokenizer(preserve_case=False,strip_handles=True)

s = "@USER @USER @USER   ;) Keep An Eye Out	NOT"
s = demoji.replace(s, "EMJ")
s = tk.tokenize(s)
print(' '.join(s))
