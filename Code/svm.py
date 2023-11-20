from sklearn.svm import SVC

def train(X, y):
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X, y)
    return clf