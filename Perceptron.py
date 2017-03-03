import numpy as np


class Perceptron:
    def __init__(self, outputter, num_features=2, alpha=0.1):
        self.alpha = alpha
        self.outputter = outputter
        self.weights = np.zeros(num_features + 1)
        self.labelset = None

    def predict_all(self, data: np.array, labels: np.array):
        right, wrong = 0, 0
        for row, expected in zip(data, labels):
            actual = self.predict(row)
            if actual != expected:
                wrong += 1
            else:
                right += 1
        return right, wrong

    def predict(self, data: np.array):
        dot = np.dot(data, self.weights)
        if dot > 0:
            return self.labelset[1]
        else:
            return self.labelset[0]

    def fit_sample(self, training_sample: np.array, desired_value):
        prediction= self.predict(training_sample)
        if prediction > desired_value:
            self.weights -= training_sample
        elif prediction < desired_value:
            self.weights += training_sample

    def run(self, training_set: np.array, d: np.array):
        self.labelset = list(set(d))
        self.labelset.sort()
        for (x, y) in zip(training_set, d):
            self.fit_sample(x, y)
        self.outputter.process(self, training_set, d, self.labelset)

    def score(self, data, labels):
        r, w = self.predict_all(data, labels)
        return (r/(r+w)) * 100