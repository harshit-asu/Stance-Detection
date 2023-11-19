"""

Data Preprocessing Steps:
1. Remove URLs
2. Remove mentions
3. Replace hashtags
4. Remove special characters
5. Remove multilple spaces
6. Correct spellings

"""

import re
from nltk.tokenize import word_tokenize

# Step 1: Remove URLs
def remove_urls(tweet):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', tweet)

# Step 2: Remove mentions
def remove_mentions(tweet, english_dictionary):
    words = tweet.split()
    for i, word in enumerate(words):
        if (word.startswith("@") or word.startswith(".@") or word.startswith("/@")) and word[1:] not in english_dictionary:
            words[i] = ""
        elif (word.startswith("@") or word.startswith(".@") or word.startswith("/@")) and word[1:] in english_dictionary:
            words[i] = word[1:]
    return " ".join(words)

# Step 3: Replace hashtags
def replace_hashtags(tweet):
    words = tweet.split()
    for i, word in enumerate(words):
        if "#" in word:
            words[i] = word.replace("#", "")
    return " ".join(words)

# Step 4: Remove special characters
def remove_special_characters(tweet):
    return re.sub(r"[^\w\s]", "", tweet)

# Step 5: Remove multiple spaces
def remove_multiple_spaces(tweet):
    return re.sub("\s+", " ", tweet)

def load_en_dict():
    with open("../Data/english_dictionary.txt") as f:
        return  [line.strip("\n") for line in f.readlines()]

def preprocess(tweets):
    # 1. URLs
    tweets = [*map(remove_urls, tweets)]
    # 2. Mentions
    english_dictionary = load_en_dict()
    tweets = [*map(remove_mentions, tweets, english_dictionary)]
    # 3. Hashtags
    tweets = [*map(replace_hashtags, tweets)]
    # 4. Special Characters
    tweets = [*map(remove_special_characters, tweets)]
    # 5. Multiple Spaces
    tweets = [*map(remove_multiple_spaces, tweets)]
    
    return tweets
