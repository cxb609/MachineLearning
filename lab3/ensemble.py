import pickle
import math
import numpy as np


class AdaBoostClassifier:

    def __init__(self, weak_classifier, n_weakers_limit):
        self.weak_classifier = weak_classifier
        self.n_weekers_limit = n_weakers_limit
        self.classifierlist = []
        self.Weight = []
        self.subclassifier = []
        pass



    def fit(self,X,y):

        m,n = X.shape
        for i in range(m):
            self.Weight.append(1/m)
        for T in range(self.n_weekers_limit):
            print(self.Weight)
            self.Weight = np.array(self.Weight)
            classifiers = self.weak_classifier(max_depth = 3)
            classifiers.fit(X,y,sample_weight = self.Weight)
            self.classifierlist.append(classifiers)
            predict = classifiers.predict(X)
            predict = predict.reshape([m, 1])
            count1 = 0
            for l in range(m):
                if predict[l] != y[l]:
                    count1 += 1
            print(count1)
            epsilon = 1 - np.sum(self.Weight.reshape([m, 1]) * predict * y)
            print(epsilon)
            e = 0
            count = 0
            for j in range(m):
                if predict[j] != y[j]:
                    count += 1
                    e = e + self.Weight[j]
            print(count)
            print(e)
            if e == 0:
                self.subclassifier.append(80)
            else:
                if e<0.5:
                    self.subclassifier.append(1/2 * math.log((1-e)/e))
                else:
                    self.subclassifier.append(0)
                    break
            for k in range(m):
                if predict[k] != y[k]:
                    self.Weight[k] = self.Weight[k]/(2*e)
                else:
                    self.Weight[k] = self.Weight[k]/(2*(1-e))
        print("the number of trees: ",T+1)
        pass


    def predict_scores(self, X):

        h = np.zeros((X.shape[0],1))
        count = 0
        for classifier in self.classifierlist:
            pred = classifier.predict(X)
            for i in range(X.shape[0]):
                h[i] += self.subclassifier[count]*pred[i]
            count += 1
        return h


        pass

    def predict(self, X, threshold=0):
        h = self.predict_scores(X)
        h = np.where(h>threshold,1,-1)
        return h
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
