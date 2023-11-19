import pandas as pd
import preprocess
import data_encoding
import svm

def preprocess_data(file_name):
    raw_data = pd.read_csv(file_name)
    preprocessed_data = raw_data.drop(["ID", "Opinion towards", "Sentiment"], axis=1)
    preprocessed_data["Tweet"] = preprocess.preprocess(list(preprocessed_data["Tweet"]))
    return preprocessed_data

def prepare_data_for_svm(data):
    targets = set(data["Target"])
    datasets = {}
    for target in targets:
        df = data[data["Target"] == target]
        favor_or_against = df[df["Stance"].isin(["FAVOR","AGAINST"])]
        datasets[target] = {
            "stance_or_none": df.replace({"Stance": {"NONE": 0, "FAVOR": 1, "AGAINST": 1}}),
            "favor_or_against": favor_or_against.replace({"Stance": {"FAVOR": 0, "AGAINST": 1}})
        }
    return datasets

def main():
    # data preprocessing
    train_data = preprocess_data(file_name="..\Data\\train_data.csv")
    # train_data.to_csv("..\Data\\train_preprocessed.csv", index=False)
    # data encoding
    train_data["Tweet"] = data_encoding.encode(train_data["Tweet"])
    # train_data.to_csv("..\Data\\train_encoded.csv", index=False)
    # prepare data for svm and lstm
    train_datasets_for_swm = prepare_data_for_svm(train_data)
    curr_data = train_datasets_for_swm["Feminist Movement"]["stance_or_none"]
    clf = svm.train(curr_data.iloc[:, 1:2], curr_data.iloc[:, -1])



if __name__ == "__main__":
    main()