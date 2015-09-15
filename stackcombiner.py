import numpy as np
from sklearn.linear_model import LogisticRegression

import util

class StackCombiner():

    def __init__(self, num_classes, C, combine_method="ovr"):
        self.num_classes = num_classes
        self.C = C
        self.combine_method = combine_method

    def fit(self, X, y):
        # print y
        N,K = X.shape
        if self.combine_method == "ovr":
            self.model = LogisticRegression(C=self.C)
            self.model.fit(X,y)
        elif self.combine_method == "ind_class":
            self.models = [LogisticRegression(C=self.C) for i in range(self.num_classes)]
            for i,m in enumerate(self.models):
                # print range(i,K,self.num_classes)
                # print X[:,range(i,K,self.num_classes)]
                m.fit(X[:,range(i,K,self.num_classes)], y[:,i])
                # print(i)
                # print(m.coef_)
                # print(m.raw_coef_)
        elif self.combine_method == 'tied_ovr':
            n_learners = X.shape[1] / self.num_classes
            combined_X = np.zeros((N * self.num_classes, n_learners))
            combined_y = np.zeros((N * self.num_classes,))
            for i in range(self.num_classes):
                # print((i * N), ((i + 1) * N))
                # print(range(i, X.shape[0], self.num_classes))
                combined_X[(i * N):((i + 1) * N), :] = X[:, range(i, X.shape[1], self.num_classes)]
                combined_y[(i * N):((i + 1) * N)] = y[:, i]
            self.model = LogisticRegression(C=self.C)
            self.model.fit(combined_X, combined_y)

    def predict_proba(self, X):
        N,K = X.shape
        if self.combine_method == "ovr":
            return self.model.predict_proba(X)
        elif self.combine_method == "ind_class":
            out = np.zeros((N,self.num_classes))
            for i,m in enumerate(self.models):
                out[:,i] = m.predict_proba(X[:,range(i,K,self.num_classes)])[:,1]

            out = out / np.reshape(np.sum(out, axis=1), (N, 1))
            return out
        elif self.combine_method == 'tied_ovr':
            n_learners = X.shape[1] / self.num_classes
            combined_X = np.zeros((N * self.num_classes, n_learners))
            for i in range(self.num_classes):
                combined_X[(i * N):((i + 1) * N), :] = X[:, range(i, X.shape[1], self.num_classes)]
            combined_y = self.model.predict_proba(combined_X)[:, 1]
            # print(N)
            # print(self.num_classes)
            # print(combined_y.shape)
            out = np.reshape(combined_y, (N, self.num_classes), order='F')
            out = out / np.reshape(np.sum(out, axis=1), (N, 1))
            # print X
            # print out
            return out

    @property
    def coef_(self):
        # FIXME - this will crash if non ovr method used
        return self.model.coef_

if __name__ == '__main__':
    X = np.random.random((10,6))
    y = np.random.randint(0,2,size=(10,3))
    X_test = np.random.random((5, 6))
    s = StackCombiner(3, 1, combine_method="ind_class")
    s.fit(X,y)
    print s.predict_proba(X_test)