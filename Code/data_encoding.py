from sentence_transformers import SentenceTransformer

def encode(tweets, electra=False):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    if electra:
        model = SentenceTransformer('ddobokki/electra-small-nli-sts')
    return [*map(model.encode, tweets)]