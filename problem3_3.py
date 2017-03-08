import sys

import numpy as np
from numpy.core.tests.test_scalarinherit import C
from scipy.constants import degree
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from io_handling import open_output


def import_and_scale_training_data(input_file_path, with_bias_column=False):
    raw_data = np.loadtxt(input_file_path, delimiter=',', skiprows=1)
    x_train, x_test, y_train, y_test = train_test_split(raw_data[:, [0,1]], raw_data[:, [2]].flatten(), test_size=0.4, random_state=42)
    return x_train, x_test, y_train, y_test


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
    # of.write("%s, %0.2f, %0.2f" % (alg, best_score, test_score))


def run_model(estimator, params, training_data, test_data, training_labels, test_labels, name, of):
    clf = GridSearchCV(estimator, params, n_jobs=-1)
    clf.fit(training_data, training_labels)
    bs = clf.best_score_
    test_score = clf.score(test_data, test_labels)
    report(of, name, bs, test_score)


def run_svm(kernel, C, degree, gamma, training_data, training_labels, test_data, test_labels):
    param_grid = [
        {'C': C, 'gamma': gamma, 'kernel': [kernel]},
    ]
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
    params = {
                'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                'degree': [4, 5, 6],
                'gamma': [0.1, 1],
                'kernel': ['linear']
    }
    run_model(SVC(), params, training_data, test_data, training_labels, test_labels, 'svm_linear', of)


def svm_with_polynomial_kernel(training_data, training_labels, test_data, test_labels, of):
    params = {
                'C': [0.1, 1, 3],
                'degree': [4, 5, 6],
                'gamma': [0.1, 1],
                'kernel': ['poly']
    }
    run_model(SVC(), params, training_data, test_data, training_labels, test_labels, 'svm_polynomial', of)


def svm_with_rbf_kernel(training_data, training_labels, test_data, test_labels, of):
    params = {
                'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                'gamma': [0.1, 0.5, 1, 3, 6, 10],
                'kernel': ['rbf']
    }
    run_model(SVC(), params, training_data, test_data, training_labels, test_labels, 'svm_rbf', of)


def logistic_regression(training_data, training_labels, test_data, test_labels, of):
    params = {
                'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                'gamma': [0.1, 0.5, 1, 3, 6, 10],
                'solver': 'liblinear'
    }
    run_model(LogisticRegression(), params, training_data, test_data, training_labels, test_labels, 'logistic', of)



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
    training_data, test_data, training_labels, test_labels = import_and_scale_training_data(sys.argv[1])
    # showGraph(data, labels)
    of = open_output(sys.argv[2])
    funcs = [svm_with_linear_kernel, svm_with_polynomial_kernel, svm_with_rbf_kernel, logistic_regression,
               k_nearest_neighbors, decision_trees, random_forest]
    wip = [logistic_regression]
    for fn in wip:
        fn(training_data, training_labels, test_data, test_labels, of)
    of.close()

if __name__ == "__main__":
    main()
