import numpy as np


class LinearRegressor:
    def __init__(self, num_features=2, iterations=100, alpha=0.0001, of=None):
        self.out = of
        self.num_features = num_features + 1  # remember to include the intercept...
        self.alpha = alpha
        self.iterations = iterations
        self.weights = np.zeros(self.num_features)

    def predict(self, data):  # given alpha row, what is the predicted value
        return np.dot(data, self.weights)

    def fit(self, data, expectations):
        e = 0.000001  # minimum change needed to declare convergence
        for i in range(self.iterations):
            v = self.weights.copy()
            self.fit_incremental_gradient_descent(data, expectations)
            total_adj = sum([abs(wi - vi) for wi, vi in zip(self.weights, v)])
            if total_adj < e:
                break
        self.out.write("%0.4f, %d, %0.4f, %0.4f, %0.4f\n" % (
        self.alpha, self.iterations, self.weights[0], self.weights[1], self.weights[2]))

    def fit_incremental_gradient_descent(self, X: np.ndarray, Y: np.ndarray):
        w = self.weights;
        h = self.predict  # abbreviations
        n = w.shape[0]
        for i, x_i in enumerate(X):  # for each x_i in X (the training data set)
            for j in range(n):  # for each feature in the test data instance, x_i,j
                w[j] -= self.adj_weight(Y[i], h(x_i), x_i[j], self.alpha)

    @staticmethod
    def adj_weight(y, hx, xij, a):
        # get a proportional fraction of the feature and remove from the corresponding weight
        try:
            return a * xij * (hx - y)
        except OverflowError:
            return 0.0
