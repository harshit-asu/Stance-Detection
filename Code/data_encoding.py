from sentence_transformers import SentenceTransformer

def encode(tweets):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    return model.encode(tweets)