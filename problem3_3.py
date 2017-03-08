import sys
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier

from io_handling import open_output


def import_and_scale_training_data(input_file_path, with_bias_column=True):
    raw_data = np.loadtxt(input_file_path, delimiter=',', skiprows=1)
    scale_data = False

    A = raw_data[:, [0]]
    B = raw_data[:, [1]]
    training_labels = raw_data[:, [2]].flatten()
    if scale_data:
        A = scale(A)
        B = scale(B)
        training_labels = scale(training_labels).flatten()

    if with_bias_column:
        rows = raw_data.shape[0]
        intercept_column = np.ones(rows)
        intercept_column.shape = (rows, 1)
        training_data = np.hstack((intercept_column, A, B))
    else:
        training_data = np.hstack((A, B))

    test_data = training_data
    test_labels = training_labels
    return training_data, training_labels, test_data, test_labels


def showGraph(data, l):
    import matplotlib.pyplot as plt

    colormap = np.array(['r', 'k'])
    labs = [int(x) for x in l]
    plt.scatter(data[:, [1]], data[:, [2]], c=colormap[labs], s=20)
    plt.show()

def measure_fitness(classifier, test_data, expectations):
    classifier.decision_function(test_data)
    predicted = classifier.predict(test_data)
    score = 0
    for x, a in zip(expectations, predicted):
        if x == a:
            score += 1
    return score


def report(of, alg, best_score, test_score):
    print("%s, %0.2f, %0.2f" % (alg, best_score, test_score))

def run_svm(kernel, C, degree, gamma, training_data, training_labels, test_data, test_labels):
    best_score = 0
    for c in C:
        for d in degree:
            for g in gamma:
                clf = svm.SVC(C=c, degree=d, gamma=g, kernel=kernel)
                clf.fit(training_data, training_labels)
                score = clf.score(test_data, test_labels)
                best_score = max(best_score, score)
    return best_score


def svm_with_linear_kernel(training_data, training_labels, test_data, test_labels, of):
    C = [0.1, 0.5, 1, 5, 10, 50, 100]
    degree = [4, 5, 6]
    gamma = [0.1, 1]
    bs = run_svm('linear', C, degree, gamma, training_data, training_labels, test_data, test_labels)
    report(of, 'svm_linear', bs, 0)


def svm_with_polynomial_kernel(training_data, training_labels, test_data, test_labels, of):
    C = [0.1, 1, 3]
    degree = [4, 5, 6]
    gamma = [0.1, 1]
    bs = run_svm('poly', C, degree, gamma, training_data, training_labels, test_data, test_labels)
    report(of, 'svm_polynomial', bs, 0)


def svm_with_rbf_kernel(training_data, training_labels, test_data, test_labels, of):
    C = [0.1, 0.5, 1, 5, 10, 50, 100]
    degree = [1]
    gamma = [0.1, 0.5, 1, 3, 6, 10]
    bs = run_svm('rbf', C, degree, gamma, training_data, training_labels, test_data, test_labels)
    report(of, 'svm_rbf', bs, 0)

def logistic_regression(training_data, training_labels, test_data, test_labels, of):
    C = [0.1, 0.5, 1, 5, 10, 50, 100]
    best_score = 0
    for c in C:
        clf = LogisticRegression(C=c, solver='liblinear')
        clf.fit(training_data, training_labels)
        score = clf.score(test_data, test_labels)
        best_score = max(best_score, score)
    report(of, 'logistic', best_score, 0)


def k_nearest_neighbors(training_data, training_labels, test_data, test_labels, of):
    best_score = 0
    for n_neighbours in range(1, 51):
        for leaf_size in range(5, 65, 5):
            clf = KNeighborsClassifier(algorithm = 'auto', n_neighbors=n_neighbours, leaf_size=leaf_size)
            clf.fit(training_data, training_labels)
            score = clf.score(test_data, test_labels)
            best_score = max(best_score, score)
    report(of, 'knn', best_score, 0)


def decision_trees(training_data, training_labels, test_data, test_labels, of):
    best_score = 0
    for max_depth in range(1, 51):
        for min_samples_split in range(2, 11):
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            clf.fit(training_data, training_labels)
            score = clf.score(test_data, test_labels)
            best_score = max(best_score, score)
    report(of, 'decision_tree', best_score, 0)



def random_forest(training_data, training_labels, test_data, test_labels, of):
    best_score = 0
    for max_depth in range(1, 51):
        for min_samples_split in range(2, 11):
            clf = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            clf.fit(training_data, training_labels)
            score = clf.score(test_data, test_labels)
            best_score = max(best_score, score)
    report(of, 'random_forest', best_score, 0)


def main():
    training_data, training_labels, test_data, test_labels = import_and_scale_training_data(sys.argv[1])
    # showGraph(data, labels)
    of = open_output(sys.argv[2])
    funcs = [svm_with_linear_kernel, svm_with_polynomial_kernel, svm_with_rbf_kernel, logistic_regression,
               k_nearest_neighbors, decision_trees, random_forest]
    wip = [k_nearest_neighbors, decision_trees, random_forest]
    for fn in funcs:
        fn(training_data, training_labels, test_data, test_labels, of)
    of.close()


if __name__ == "__main__":
    main()
