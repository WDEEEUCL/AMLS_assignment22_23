from os.path import join

path = 'C:/Users/Wei Dai/PycharmProjects/AMLS_22-23_SN19111862/A1/landmarks_HOG_A1.py'

from importlib.machinery import SourceFileLoader
somemodule = SourceFileLoader('landmarks_HOG_A1', join(path)).load_module()

import landmarks_HOG_A1 as l2
import numpy as np

from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm


def get_data():
    X, y = l2.extract_features_labels()

    Y = np.array([y, -(y - 1)]).T

    tr_X = X[:4000]
    tr_Y = Y[:4000]
    te_X = X[4000:]
    te_Y = Y[4000:]

    return tr_X, tr_Y, te_X, te_Y


# sklearn functions implementation
def img_SVM(training_images, training_labels, test_images, test_labels):
    #classifier = svm.SVC(kernel='linear')
    classifier = svm.SVC(kernel='poly')

    classifier.fit(training_images, training_labels)

    pred = classifier.predict(test_images)

    print(pred)

    print("Accuracy:", accuracy_score(test_labels, pred))

tr_X, tr_Y, te_X, te_Y= get_data()

pred=img_SVM(tr_X.reshape((4000, 68*2)), list(zip(*tr_Y))[0], te_X.reshape((len(te_X), 68*2)), list(zip(*te_Y))[0])