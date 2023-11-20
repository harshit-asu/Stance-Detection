from sklearn.metrics import accuracy_score, classification_report
import preprocess
import data_encoding
import json
from joblib import load
import numpy as np

def evaluate(model, X, y):
    y_pred = predict(model, X)
    report = classification_report(y, y_pred, output_dict=True)
    return report

def predict(model, X):
    return model.predict(X)

def determine_classifier(target, model):
    with open("..\Data\\targets.json", "r") as f:
        targets_json = json.load(f)
    base_folder = "lstm"
    if model == "svm" or model == "svm_electra":
        base_folder = "svm"
    elif model == "lstm" or model == "lstm_electra":
        base_folder = "lstm"
    clf1 = load(f"models\{base_folder}\{model}_{targets_json[target]}_stance_or_none.joblib")
    clf2 = load(f"models\{base_folder}\{model}_{targets_json[target]}_favor_or_against.joblib")
    return clf1, clf2


def process_web_input(tweet, target, model):
    electra = False
    if model == "svm_electra" or model == "lstm_electra":
        electra = True
    cleaned_tweet = preprocess.preprocess([tweet])[0]
    encoded_tweet = np.array(data_encoding.encode([cleaned_tweet], electra))
    clf1, clf2 = determine_classifier(target, model)
    y1 = predict(clf1, encoded_tweet)
    if y1 == 0:
        return "NEUTRAL", "black"
    else:
        y2 = predict(clf2, encoded_tweet)
        if y2 == 1:
            return "AGAINST", "red"
        else:
            return "FAVOR", "green"