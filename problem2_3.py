import sys
from matplotlib import pyplot as plt

import numpy as np
import sklearn.preprocessing
from numpy.distutils.system_info import p
from sklearn.preprocessing import scale

import LinearRegression


def import_and_scale_training_data(input_file_path, with_bias_column=True):
    raw_data = np.loadtxt(input_file_path, delimiter=',')
    scale_data = False

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
    lr = LinearRegression.LinearRegressor(iterations=100, alpha=0.001)
    lr.fit(data, heights)
    print("Right=%d\t Wrong=%d\tAbove=%d\tBelow=%d" % lr.predict_all(data, heights))
    print("weights=%s" % str(lr.weights))
    showGraph(data, heights, lr.weights)


if __name__ == "__main__":
    main()

# PROBLEM:
# - The Linear Regression Algorithm doesn't predict values at all
#
# KNOWN:
# - the weights are being updated
# - the data is being scaled as requested
# - some predictions are above, and some are below the expected figure:
#     Right=0	 Wrong=79	Above=37	Below=42
# - the solution splits the data right down the middle in terms of being above or below the expected values.
#       - ? should that imply something about the kind of function I've got? i.e. is it centering one variable but not the other?
# - places where the alg could go wrong:
#      - data ingestion
#           - checking by hand confirms this is right
#      - data scaling and prep
#           - did this by hand and via the np.scale function and both cam out same
#      - with the logic of the alg
#           - I could have misunderstood the alg, by not understanding the maths notation
#               - e.g The problem notes reference x_i in the gradient desc eq, but Andrew Ng has x^i_j (i.e. the jth feature of x_i)
#                   - the problem notes contain an error somewhere.
#                   - Russel and Norvig has x_{j,i} which agrees with Ng.
#                       - w_i := w_i + \alpha * \Sum_jx_{j,i}(y_j - h_w(x_j))
#                       - This is summing the product of the feature x_{j,i} with the loss, and multiplying the result with alpha.
#                       - should it matter whether the product is calculated after or before the sum?
#      - with the impl of the alg
#      - with the logic of the predictions
#      - with the impl of the predictions
#      - with judging whether the prediction is right or wrong
# - The weights after (100, 0.001) are:
#       weights=[  1.24644739e-15   8.29643668e-01   5.58111679e-02]
#           the intercept is effectively 0
#           age ~= 0.8
#           weight ~= 0.05
#           implies that height is mostly dependent on age rather than weight.
# - The plane defined by the weights transects the middle of the points.  Is that because the points have been scaled
#   to be zero, or because it is reflecting the data t least in that dimension?
#
# QUESTIONS:
# - is the prediction always more, or always less, than the expected value?
# - am I misunderstanding the problem?  Is it just that the tolerance for correctness iis too high,
#    and that the data is too variable?
# - is it a problem with the way i am calculating the predicion, or with how I am updating the weights?
# - What are the places where this could go wrong?
# - Should the fact that equal parts above and below the prediction imply something about the kind of function
#       I've got? i.e. is it centering one variable but not the other?
# - Is there anything interesting about the equation formed by the weights?
#
# TESTS:
# - Check whether the miss is always below or above the desired value
# - examine the weights to see what kind of function they compute.
# - stop scaling the data, and see how it behaves then.
#       - seemed that the weight alterations increased in ever larger increments.
# - Try the non-batch gradient descent version of the algorithm