import numpy as np
from matplotlib import pyplot as plt

from Perceptron import Perceptron


def open_input(input_file):
    fo = open(input_file, "r")
    return fo


def open_output(output_file):
    fo = open(output_file, "w")
    return fo


class Outputter:
    def process(self, p: Perceptron, data: np.array, expected_labels: np.array, labels: np.array):
        pass


class CompositeOutputter(Outputter):
    def __init__(self, outputters):
        super().__init__()
        self.outputters = outputters

    def process(self, p: Perceptron, data: np.array, expected_labels: np.array, labels: np.array):
        for outputter in self.outputters:
            outputter.process(p, data, labels)


class ConsoleOutputter(Outputter):
    def process(self, p: Perceptron, data: np.array, expected_labels: np.array, labels: np.array):
        print(p.weights)
        # print("Prediction " + str(p.predict(data)))
        # print("Actual     " + str(labels))
        # print("Accuracy   " + str(p.score(data, labels) * 100) + "%")

    def __init__(self):
        super().__init__()


class FileOutputter(Outputter):
    def process(self, p: Perceptron, data: np.array, expected_labels: np.array, labels: np.array):
        self.fo.write("%d, %d, %d\n"%(p.weights[1], p.weights[2], p.weights[0]))

    def __init__(self, file_path):
        super().__init__()
        self.fo = open_output(file_path)


class GraphOutputter(Outputter):
    def __init__(self):
        super().__init__()

    def process(self, p: Perceptron, data: np.array, expected_labels: np.array, labels: np.array):
        colormap = np.array(['r', 'k'])
        ixs = [0 if x == 1 else 1 for x in expected_labels]
        xs = data[:, [1]]
        ys = data[:, [2]]
        plt.scatter(xs.flatten(), ys.flatten(), c=colormap[ixs])
        w = p.weights
        xx = np.linspace(min(xs), max(xs))
        a = -w[1] / w[2]
        yy = a * xx - (w[0]) / w[2]
        plt.plot(xx, yy, 'k-')
        plt.show()