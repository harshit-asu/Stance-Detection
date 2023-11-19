import pandas as pd

# training data
with open('..\Data\\raw_data\\trainingdata-all-annotations.txt', "r") as f:
    lines = f.readlines()
    cols = lines[0].strip().split('\t')
    df = pd.DataFrame([], columns=cols)
    for i in range(1, len(lines)):
        df.loc[len(df)] = lines[i].strip().split('\t')
    df.to_csv("..\Data\\train_data.csv", index=False)

# test data
with open('..\Data\\raw_data\\testdata-taskA-all-annotations.txt', "r") as f:
    lines = f.readlines()
    cols = lines[0].strip().split('\t')
    df = pd.DataFrame([], columns=cols)
    for i in range(1, len(lines)):
        df.loc[len(df)] = lines[i].strip().split('\t')
    df.to_csv("..\Data\\test_data.csv", index=False)