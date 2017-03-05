import sys

import numpy as np

from Perceptron import Perceptron
from io_handling import FileOutputter


def main():
    p = Perceptron(FileOutputter(sys.argv[2]))
    raw_data = np.loadtxt(sys.argv[1], delimiter=',')
    data = raw_data[:, [0, 1]]
    rows = raw_data.shape[0]
    bias_column = np.ones(rows)
    bias_column.shape = (rows, 1)
    data = np.hstack((bias_column, data))
    labels = raw_data[:, [2]].flatten()


    while True:
        p.run(data, labels.T)
        r,w = p.predict_all(data, labels.T)
        if w == 0:
            break

    return 0


if __name__ == "__main__":
    main()

# # RESOURCES
# - https://www.tutorialspoint.com/python/python_files_io.htm (file handling)
# - https://en.wikipedia.org/wiki/Perceptron (WP article on perceptron)
# - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron (basic perceptron learning algorithm)
# - https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html (loading text from csv into numpy array)
