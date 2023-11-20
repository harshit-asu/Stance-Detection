import pandas as pd
import numpy as np
import preprocess
import data_encoding
import svm
import lstm
import json
from joblib import dump, load
import test_utils

def preprocess_data(file_name):
    raw_data = pd.read_csv(file_name)
    preprocessed_data = raw_data.drop(["ID", "Opinion towards", "Sentiment"], axis=1)
    preprocessed_data["Tweet"] = preprocess.preprocess(list(preprocessed_data["Tweet"]))
    return preprocessed_data

def prepare_data_for_svm(train_targets, train_embeddings, train_stances, test_targets, test_embeddings, test_stances):
    targets = set(train_targets)
    train_datasets = {}
    test_datasets = {}
    for target in targets:
        ind_train = list(train_targets[train_targets == target].index)
        ind_test = list(test_targets[test_targets == target].index)
        ind_train_fa = list(train_stances[ind_train][train_stances.isin(["FAVOR", "AGAINST"])].index)
        ind_test_fa = list(test_stances[ind_test][test_stances.isin(["FAVOR", "AGAINST"])].index)
        train_datasets[target] = {
            "stance_or_none": {
                "X": train_embeddings[ind_train],
                "y": train_stances[ind_train].replace({"NONE": 0, "FAVOR": 1, "AGAINST": 1})
            },
            "favor_or_against": {
                "X": train_embeddings[ind_train_fa],
                "y": train_stances[ind_train_fa].replace({"FAVOR": 0, "AGAINST": 1})
            }
        }
        test_datasets[target] = {
            "stance_or_none": {
                "X": test_embeddings[ind_test],
                "y": test_stances[ind_test].replace({"NONE": 0, "FAVOR": 1, "AGAINST": 1})
            },
            "favor_or_against": {
                "X": test_embeddings[ind_test_fa],
                "y": test_stances[ind_test_fa].replace({"FAVOR": 0, "AGAINST": 1})
            }
        }
    return train_datasets, test_datasets

def data_preparation(train_file, test_file):
    # data preprocessing
    train_data = preprocess_data(file_name=train_file)
    # data encoding
    train_data["Tweet"] = data_encoding.encode(train_data["Tweet"])
    # train_data.to_csv("..\Data\\train_encoded.csv", index=False)
    train_embeddings = np.array(list(train_data["Tweet"]))
    # data preprocessing: test data
    test_data = preprocess_data(file_name=test_file)
    # data encoding
    test_data["Tweet"] = data_encoding.encode(test_data["Tweet"])
    test_embeddings = np.array(list(test_data["Tweet"]))
    # test_data.to_csv("..\Data\\test_encoded.csv", index=False)
    np.save("..\Data\\train_embeddings.npy", train_embeddings)
    np.save("..\Data\\test_embeddings.npy", test_embeddings)

def load_data(electra=False):
    suffix = "embeddings"
    if electra:
        suffix = "embeddings_electra"
    train_data = pd.read_csv("..\Data\\train_data.csv")
    test_data = pd.read_csv("..\Data\\test_data.csv")
    train_embeddings = np.load(f"..\Data\\train_{suffix}.npy")
    test_embeddings = np.load(f"..\Data\\test_{suffix}.npy")
    return train_data["Target"], train_embeddings, train_data["Stance"], test_data["Target"], test_embeddings, test_data["Stance"]

def train_svm_classifiers(train_datasets_for_svm, test_datasets_for_svm, electra=False):
    results = {}
    prefix = "svm"
    if electra:
        prefix = "svm_electra"
    with open("..\Data\\targets.json", "r") as f:
        targets_json = json.load(f)
    print("\n********* Support Vector Machines *********")
    for target, datasets in train_datasets_for_svm.items():
        results[target] = {}
        print("\n\n\nTarget: {}".format(target))
        for classifier_type in datasets.keys():
            results[target][classifier_type] = {}
            print("\n\nClassifier type: {}".format(classifier_type))
            X_train, y_train = datasets[classifier_type].values()
            clf = svm.train(X_train, y_train)
            # save the classifier
            dump(clf, f"models\svm\{prefix}_{targets_json[target]}_{classifier_type}.joblib")
            # load again
            clf = load(f"models\svm\{prefix}_{targets_json[target]}_{classifier_type}.joblib")
            # training report
            train_report = test_utils.evaluate(clf, X_train, y_train)
            results[target][classifier_type]["train"] = train_report
            print("\nTraining Accuracy: {}".format(train_report["accuracy"]))
            # test report
            X_test, y_test = test_datasets_for_svm[target][classifier_type].values()
            test_report = test_utils.evaluate(clf, X_test, y_test)
            results[target][classifier_type]["test"] = test_report
            print("\nTest Accuracy: {}".format(test_report["accuracy"]))
    print("\n*******************************************\n")
    with open(f"..\Evaluations\{prefix}_results.json", "w") as f:
        # json_string = json.dumps(results)
        json.dump(results, f)

def train_lstm_models(train_datasets, test_datasets, electra=False):
    results = {}
    prefix = "lstm"
    if electra:
        prefix = "lstm_electra"
    with open("..\Data\\targets.json", "r") as f:
        targets_json = json.load(f)
    print("\n********* LSTM *********")
    for target, datasets in train_datasets.items():
        results[target] = {}
        print("\n\n\nTarget: {}".format(target))
        for classifier_type in datasets.keys():
            results[target][classifier_type] = {}
            print("\n\nClassifier type: {}".format(classifier_type))
            X_train, y_train = datasets[classifier_type].values()
            clf = lstm.train(X_train, y_train)
            # save the classifier
            dump(clf, f"models\lstm\{prefix}_{targets_json[target]}_{classifier_type}.joblib")
            # load again
            clf = load(f"models\lstm\{prefix}_{targets_json[target]}_{classifier_type}.joblib")
            # training report
            train_report = test_utils.evaluate(clf, X_train, y_train)
            results[target][classifier_type]["train"] = train_report
            print("\nTraining Accuracy: {}".format(train_report["accuracy"]))
            # test report
            X_test, y_test = test_datasets[target][classifier_type].values()
            test_report = test_utils.evaluate(clf, X_test, y_test)
            results[target][classifier_type]["test"] = test_report
            print("\nTest Accuracy: {}".format(test_report["accuracy"]))
    print("\n*******************************************\n")
    with open(f"..\Evaluations\{prefix}_results.json", "w") as f:
        # json_string = json.dumps(results)
        json.dump(results, f)

def main():
    # run this function only for the first time
    # data_preparation(train_file="..\Data\\train_data.csv", test_file="..\Data\\test_data.csv")
    # prepare data for svm and lstm
    electra = True
    train_targets, train_embeddings, train_stances, test_targets, test_embeddings, test_stances = load_data(electra)
    train_datasets_for_svm, test_datasets_for_svm = prepare_data_for_svm(train_targets, train_embeddings, train_stances, test_targets, test_embeddings, test_stances)
    # svm classifiers - only once
    # train_svm_classifiers(train_datasets_for_svm, test_datasets_for_svm, electra=True)
    # LSTM classifiers - only once
    # train_lstm_models(train_datasets_for_svm, test_datasets_for_svm, electra)


if __name__ == "__main__":
    main()