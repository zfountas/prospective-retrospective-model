from sklearn.svm import SVR
import numpy as np
from os.path import isfile
import pickle

class Regressor:
    def __init__(self, filename = ""):
        self.training = True
        # very good results: (kernel='rbf', C=1e3, gamma=0.0001)
        self.filename = filename
        if self.filename != "" and isfile(self.filename):
            self.regressor = pickle.load(open(self.filename, 'rb'))
            self.trained = True
        else:
            self.regressor = SVR(kernel='rbf', C=1e3, gamma=0.0001)
            self.trained = False

    def train(self, train_x, train_y):
        #print np.array(train_x)
        # Fit regression model
        self.regressor.fit(np.array(train_x), np.array(train_y))
        if self.filename != "":
            with open(self.filename, 'wb') as ff:
                pickle.dump(self.regressor, ff)
        self.trained = True

    def predict(self, test_x):
        if len(np.shape(np.array(test_x))) == 1:
            estimate = self.regressor.predict(np.array(test_x).reshape(1, -1))
        else:
            estimate = self.regressor.predict(np.array(test_x))
            #print "TIME:", estimate
        return estimate
