import numpy as np
from scipy.special import expit


class LinearRegressor:
    def __init__(self, num_features=2, iterations=100, alpha=0.0001, of=None):
        self.out = of
        self.num_features = num_features + 1  # remember to include the intercept...
        self.alpha = alpha
        self.iterations = iterations
        self.weights = np.zeros(self.num_features)
        self.use_gradient_descent = True

    def predict_all(self, data, expectations, threshold=0.01):
        right, wrong, below, above = 0, 0, 0, 0
        for row, expected in zip(data, expectations):
            actual = self.predict(row)
            if abs(actual - expected) < threshold:
                right += 1
            else:
                wrong += 1

            if actual < expected:
                below += 1
            elif actual > expected:
                above += 1
        return right, wrong, below, above

    def predict(self, data) -> float:  # given alpha row, what is the predicted value
        return np.dot(data, self.weights)

    def fit(self, data, expectations):
        e = 0.000001  # minimum change needed to declare convergence
        for i in range(self.iterations):
            v = self.weights.copy()

            if self.use_gradient_descent:
                self.fit_incremental_gradient_descent(data, expectations)
            else:
                self.fit_analytic(data, expectations)

            total_adj = sum([abs(wi - vi) for wi, vi in zip(self.weights, v)])
            if total_adj < e:
                # declare convergence!
                # print("converged after %d iterations" % i)
                break
            # after each cycle test the predictions against the learning data
            # r, w, a, b = self.predict_all(data, expectations)
            # if i % (self.iterations / 10) == 0:
            #     print("Iteration %d: Right=%d\t Wrong=%d" % (i, r, w))
            # if w == 0:
            #     break
        self.out.write("%0.3f, %d, %0.4f, %0.4f, %0.4f\n"%(self.alpha, self.iterations, self.weights[0], self.weights[1], self.weights[2]))
    # derived from Russel and Norwig, pp. 721, eq. 18.6
    def fit_gradient_descent_old(self, X: np.ndarray, Y: np.ndarray):
        e = 0.000001  # minimum change needed to declare convergence
        n = X.shape[0]  # how many rows do we have to work through
        W = self.weights

        for i, x_i in enumerate(X):
            W_before = self.weights.copy()
            adj = self.alpha * (Y[i] - self.predict(x_i))
            for j, w_j in enumerate(self.weights):
                W[j] = w_j + (adj * x_i[j])

            total_adj = sum([abs(x - y) for x, y in zip(W, W_before)])
            if total_adj < e:
                # declare convergence!
                print("converged after %d iterations" % i)
                break

    # B = (X^T * X)^-1 * X^T * y
    def fit_analytic(self, data, expectations):
        data, expectations = np.asmatrix(data), np.asmatrix(expectations)
        self.weights = (data.T * data).I * data.T * expectations.T

    def fit_batch_gradient_descent(self, X: np.ndarray, Y: np.ndarray):
        w = self.weights
        # n = X.shape[0]  # how many rows do we have to work through

        v = self.weights.copy()  # copy of the weights prior to adjustment, to check for convergence
        for j, w_j in enumerate(self.weights):
            adj = np.mean([x_i[j] * (self.predict(x_i) - Y[i]) for i, x_i in enumerate(X)])
            w[j] = w_j - (self.alpha * adj * 0.5)

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
        return a * xij * (hx - y)
