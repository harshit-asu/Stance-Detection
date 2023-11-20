import json

svm_file = "..\Evaluations\svm_results.json"
svm_electra_file = "..\Evaluations\svm_electra_results.json"
lstm_file = "..\Evaluations\lstm_results.json"
lstm_electra_file = "..\Evaluations\lstm_electra_results.json"

with open(svm_file, "r") as f:
    svm_results = json.load(f)
    print("\n\nSVM with BERT\n")
    for target, results in svm_results.items():
        for clf_type, vals in results.items():
            for train_or_test, acc in vals.items():
                accuracy = acc["accuracy"]
                if train_or_test == "test":
                    print(f"{target}: {clf_type}: {train_or_test}: {accuracy}")
    print("\n\n")


with open(svm_electra_file, "r") as f:
    svm_results = json.load(f)
    print("\n\nSVM with ELECTRA\n")
    for target, results in svm_results.items():
        for clf_type, vals in results.items():
            for train_or_test, acc in vals.items():
                accuracy = acc["accuracy"]
                if train_or_test == "test":
                    print(f"{target}: {clf_type}: {train_or_test}: {accuracy}")
    print("\n\n")


with open(lstm_file, "r") as f:
    svm_results = json.load(f)
    print("\n\nLSTM with BERT\n")
    for target, results in svm_results.items():
        for clf_type, vals in results.items():
            for train_or_test, acc in vals.items():
                accuracy = acc["accuracy"]
                if train_or_test == "test":
                    print(f"{target}: {clf_type}: {train_or_test}: {accuracy}")
    print("\n\n")


with open(lstm_electra_file, "r") as f:
    svm_results = json.load(f)
    print("\n\nLSTM with ELECTRA\n")
    for target, results in svm_results.items():
        for clf_type, vals in results.items():
            for train_or_test, acc in vals.items():
                accuracy = acc["accuracy"]
                if train_or_test == "test":
                    print(f"{target}: {clf_type}: {train_or_test}: {accuracy}")
    print("\n\n")