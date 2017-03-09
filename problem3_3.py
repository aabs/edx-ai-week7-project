import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from io_handling import open_output


def import_and_scale_training_data(input_file_path, with_bias_column=False):
    raw_data = np.loadtxt(input_file_path, delimiter=',', skiprows=1)
    x_train, x_test, y_train, y_test = train_test_split(raw_data[:, [0, 1]], raw_data[:, [2]].flatten(), test_size=0.4,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def showGraph(test_data, test_labels, clf):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets

    # import some data to play with
    X = test_data[:, :2]  # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    y = test_labels

    h = .02  # step size in the mesh

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # plt.subplot(2, 2, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("blah")

    plt.show()

    # import matplotlib.pyplot as plt
    #
    # colormap = np.array(['r', 'k'])
    # labs = [int(x) for x in l]
    # plt.scatter(data[:, [1]], data[:, [2]], c=colormap[labs], s=20)
    # plt.show()


def report(output_stream, classifier_name, best_score, test_score):
    print("%s, %0.2f, %0.2f" % (classifier_name, best_score, test_score))
    # output_stream.write("%s, %0.2f, %0.2f" % (classifier_name, best_score, test_score))


def run_model(estimator, params, training_data, test_data, training_labels, test_labels, classifier_name,
              output_stream):
    clf = GridSearchCV(estimator, params, n_jobs=-1)
    clf.fit(training_data, training_labels)
    best_score = clf.best_score_
    test_score = clf.score(test_data, test_labels)
    report(output_stream, classifier_name, best_score, test_score)


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
        'solver': ['liblinear']
    }
    run_model(LogisticRegression(), params, training_data, test_data, training_labels, test_labels, 'logistic', of)


def k_nearest_neighbors(training_data, training_labels, test_data, test_labels, of):
    params = {
        'n_neighbors': range(1, 51),
        'leaf_size': range(5, 65, 5),
        'algorithm': ['auto']
    }
    run_model(KNeighborsClassifier(), params, training_data, test_data, training_labels, test_labels, 'knn', of)

def decision_trees(training_data, training_labels, test_data, test_labels, of):
    params = {
        'max_depth': range(1, 51),
        'min_samples_split': range(2, 11)
    }
    run_model(DecisionTreeClassifier(), params, training_data, test_data, training_labels, test_labels, 'decision_tree', of)

def random_forest(training_data, training_labels, test_data, test_labels, of):
    params = {
        'max_depth': range(1, 51),
        'min_samples_split': range(2, 11)
    }
    run_model(RandomForestClassifier(), params, training_data, test_data, training_labels, test_labels, 'random_forest', of)


def main():
    training_data, test_data, training_labels, test_labels = import_and_scale_training_data(sys.argv[1])
    # showGraph(data, labels)
    of = open_output(sys.argv[2])
    funcs = [
          svm_with_linear_kernel
        , svm_with_polynomial_kernel
        , svm_with_rbf_kernel
        , logistic_regression
        , k_nearest_neighbors
        , decision_trees
        , random_forest
    ]
    wip = [
        logistic_regression
    ]
    for fn in funcs:
        fn(training_data, training_labels, test_data, test_labels, of)
    of.close()


if __name__ == "__main__":
    main()
