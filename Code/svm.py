from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train(X, y):
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X, y)
    return clf


def evaluate(clf, x, y_true):
    y_pred = predict(clf, x)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict= True)
    print(accuracy, report)

def predict(model, x):
    return model.predict(x)