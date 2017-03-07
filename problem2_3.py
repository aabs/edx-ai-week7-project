import sys
from matplotlib import pyplot as plt

import numpy as np
import sklearn.preprocessing
from numpy.distutils.system_info import p
from sklearn.preprocessing import scale

import LinearRegression
from sklearn import linear_model

from io_handling import open_output


def import_and_scale_training_data(input_file_path, with_bias_column=True):
    raw_data = np.loadtxt(input_file_path, delimiter=',')
    scale_data = True

    age = raw_data[:, [0]]
    weight = raw_data[:, [1]]
    heights = raw_data[:, [2]].flatten()
    if scale_data:
        age = scale(raw_data[:, [0]])
        weight = scale(raw_data[:, [1]])
        heights = scale(raw_data[:, [2]]).flatten()

    if with_bias_column:
        rows = raw_data.shape[0]
        intercept_column = np.ones(rows)
        intercept_column.shape = (rows, 1)
        data = np.hstack((intercept_column, age, weight))
    else:
        data = np.hstack((age, weight))
    return data, heights


def showGraph(data, heights, weights):
    from matplotlib import pyplot
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = pylab.figure()
    ax = Axes3D(fig)
    xs = data[:, [1]].flatten()
    ys = data[:, [2]].flatten()
    zs = heights
    ax.scatter(xs, ys, zs)
    w = weights
    # xx = np.linspace(min(xs), max(xs))
    # a = -w[1] / w[2]
    # yy = a * xx - (w[0]) / w[2]
    # plt.plot(xx, yy, 'k-')

    xx, yy = np.meshgrid(np.arange(min(xs), max(xs)), np.arange(min(ys), max(ys)))

    # calculate corresponding z
    z = (-w[1] * xx - w[2] * yy - w[0]) * 1. / w[2]

    plt3d = fig.gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)

    pyplot.show()


def main():
    data, heights = import_and_scale_training_data(sys.argv[1])
    of = open_output(sys.argv[2])
    for iterations, alpha in [(100, 0.001), (100, 0.005), (100, 0.01),
                              (100, 0.05), (100, 0.1), (100, 0.5),
                              (100, 1), (100, 5), (100, 10), (1000, 0.0005)]:
        lr = LinearRegression.LinearRegressor(iterations=iterations, alpha=alpha, of=of)
        lr.fit(data, heights)
    of.close()


if __name__ == "__main__":
    main()

""""""
# PROBLEM:
# - The Linear Regression Algorithm doesn't predict values at all
#
# KNOWN:
# - !!!!! It seems to have been working for ages!,  What was wrong was how I was testing if it was right!
#   - SOB
# - when the data is unscaled, the alg ends up getting 6 right after 5k iterations. With scaled data it is 0.
#   <- the model seems to suppress the values for the most part. is it really proportionate to the error?
#   <- when I edited the scaled_data flag to and from True, the alg stop getting and positives.
# - the weights are being updated
# - the data is being scaled as requested
# - some predictions are above, and some are below the expected figure:
#     Right=0	 Wrong=79	Above=37	Below=42
# - the solution splits the data right down the middle in terms of being above or below the expected values.
#       - ? should that imply something about the kind of function I've got? i.e. is it centering one variable
#            but not the other?
# - places where the alg could go wrong:
#      - data ingestion
#           <- checking by hand confirms this is correct
#      - data scaling and prep
#           <- did this by hand and via the np.scale function and both cam out same
#      - with the logic of the alg
#           - I could have misunderstood the alg, by not understanding the maths notation
#               - e.g The problem notes reference x_i in the gradient desc eq, but Andrew Ng has x^i_j (i.e. the jth
#                   feature of x_i)
#                   <- the problem notes contain an error in notation.
#                   - Russel and Norvig has x_{j,i} which agrees with Ng.
#                       - w_i := w_i + \alpha * \Sum_jx_{j,i}(y_j - h_w(x_j))
#                       - This is summing the product of the feature x_{j,i} with the loss, and multiplying the
#                           result with alpha.
#                           - multiplying the loss value with x_ji scales the adjustment
#                             to the the size of the feature weight itself.  if the
#                             weight is small, then large adjustments would have a
#                             disproportionate effect on the behaviour - the gradient
#                             descent would not be incremental or infinitesimal, (so it
#                             makes sense to have that factor in the alg)
#      - with the impl of the alg
#           <- tried (grad desc (batch and incremental) as well as analytic.  All fail in the same way
#               ?-> is it logical to assume the error lies somewhere else?
#      - with the logic of the predictions
#      - with the impl of the predictions
#           <- implemented the inner/dot product of x*w by hand as well as through 2 std APIs.
#               All agree on the answer
#      - with judging whether the prediction is right or wrong
#           -> stop scaling the height column.  direct comparison can then be made between h(x) and f
# - The weights after (100, 0.001) are:
#       weights=[  1.24644739e-15   8.29643668e-01   5.58111679e-02]
#           the intercept is effectively 0
#           age ~= 0.8
#           weight ~= 0.05
#           implies that height is mostly dependent on age rather than weight?
# - The plane defined by the weights transects the middle of the points.  Is that because the points have been scaled
#   to cluster around the zero point, or because it is truly reflecting the data?
# - Updating the weights in one pass, so that they don't distort the ongoing calculation does not affect the outcome
# - I've paired the code back about as far as I can:
#
#         def fit_incremental_gradient_descent(self, X: np.ndarray, Y: np.ndarray):
#             w = self.weights; h = self.predict  # abbreviations
#             n = w.shape[0]
#             for i, x_i in enumerate(X):
#                 for j in range(n):
#                     w[j] -= self.alpha * (x_i[j] * (h(x_i) - Y[i]))
#
# QUESTIONS:
# - is the prediction always more, or always less, than the expected value?
# - am I misunderstanding the problem?  Is it just that the tolerance for correctness iis too high,
#    and that the data is too variable?
# - is it a problem with the way i am calculating the prediction, or with how I am updating the weights?
# - What are the places where this could go wrong?
# - Should the fact that equal parts above and below the prediction imply something about the kind of function
#       I've got? i.e. is it centering one variable but not the other?
# - Is there anything interesting about the equation formed by the weights?
# - should it matter whether the product is calculated after or before the sum?
#     <- tried it both ways. makes no difference
# - Assuming that:
#   - (1) the algorithm is implemented properly,
#   - (2) the data has been prepared properly
# - Where should I look next for issues?
#   - 3rd party APIs used without proper understanding
#   - semantics of the operators in use
#   - choice of data structures, orientations, transpositions etc
# - Do I need to scale the height column?
#
# TESTS:
# - Check whether the miss is always below or above the desired value
# - examine the weights to see what kind of function they compute.
# - stop scaling the data, and see how it behaves then.
#       - seemed that the weight alterations increased in ever larger increments.
# - Try the non-batch gradient descent version of the algorithm
#       - didn't make any difference.
#           ?->  Don't think my problem is with my impl of the alg, since several alternate
#               impls all fail in the same way.  Does that imply that there is an issue somewhere else?
# - check that the adjustments being made are proportionate to the kind of error detected:
#   - i.e. higher should result in -ve adj and vice versa
# - initialise weights with ones rather than zeros


