import nltk

nltk.download("words")
english_dictionary = set(nltk.corpus.words.words())

with open("../Data/english_dictionary.txt", "w") as f:
    f.write("\n".join(english_dictionary))