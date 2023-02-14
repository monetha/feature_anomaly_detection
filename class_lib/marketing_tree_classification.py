import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd

from scipy.sparse import csr_matrix

from math import isnan


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """
    return np.var(y)

def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    probs = y.mean(axis=0)
    return -np.sum(probs * np.log(probs + EPS))


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0, children=[], node_id=-1, 
                 total_samples=-1, cur_samples=-1):
        self.feature = {node_id: feature_index}
        self.value = {node_id: threshold}
        self.threshold = {node_id: threshold}
        self.children_left = None
        self.children_right = None
        self.left_child = None
        self.right_child = None
        self.children = children
        self.is_leaf = False
        self.node_id = node_id
        self.n_outputs = -1
        self.impurity = []
        self.n_classes = {0: total_samples}
        self.feature_index = feature_index
        self.node_count = -1
        self.features_path_str = []


class DecisionTree(BaseEstimator):
    all_criterions = {
        'variance': (variance, False),
        'entropy': (entropy, True)
    }

    def __init__(self, feature_names, n_classes=None, max_depth=100, min_samples_leaf=2,
                 criterion='variance', debug=False, c=1500.0, random_state=0, last_features=4,
                 use_layer_constraint=True):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_leaf
        self.criterion, self.classification = self.all_criterions[criterion]
        self.decicion_function = self.criterion

        self.depth = 0
        self.tree_ = None  # Use the Node class to initialize it later
        self.debug = debug
        self.c = c
        self.is_fitted_ = False
        self.node_count = 0
        self.n_samples = -1
        self.impurity = {}
        self.last_node_id = 0
        self.children_left = {}
        self.children_right = {}
        self.value = {}
        self.probas = {}
        self.feature = {}
        self.threshold = {}
        self.n_node_samples = {}
        self._estimator_type = ""

        self.last_n_features_size = last_features
        self.features_names = feature_names
        self.features_in_levels = {}
        if self.classification:
            self._estimator_type = "classifier"
        else:
            self._estimator_type = "regressor"

        self.use_layer_constraint = use_layer_constraint
        

    def get_feature_name(self, index):
        full_feature = self.feature_names[index]
        pass


    def apply(self, X):
        X_ndarray = X.copy()
        n_samples = X.shape[0]

        # Initialize output
        out = np.zeros((n_samples,), dtype=np.intp)


        for i in range(n_samples):
            node_id = 0
            while self.children_left[node_id] != -1:
                if X_ndarray[i, self.feature[node_id]] <= self.threshold[node_id]:
                    node_id = self.children_left[node_id]
                else:
                    node_id = self.children_right[node_id]

            out[i] = node_id

        return out

    def decision_path(self, X, check_input=True):
        # Extract input
        X_ndarray = X.copy()
        n_samples = X.shape[0]

        # Initialize output

        indptr = np.zeros(n_samples + 1, dtype=np.intp)
        indices = np.zeros(n_samples * (1 + self.max_depth), dtype=np.intp)

        for i in range(n_samples):
            node_id = 0
            indptr[i + 1] = indptr[i]

            while self.children_left[node_id] != -1:
                indices[indptr[i + 1]] = node_id
                indptr[i + 1] += 1

                if X_ndarray[i, self.feature[node_id]] <= self.threshold[node_id]:
                    node_id = self.children_left[node_id]
                else:
                    node_id = self.children_right[node_id]

            # Add the leave node
            indices[indptr[i + 1]] = node_id
            indptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        data = np.ones(shape=len(indices), dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        threshold_condition = X_subset[:, feature_index] <= threshold
        X_left, y_left = X_subset[threshold_condition], y_subset[threshold_condition]
        X_right, y_right = X_subset[threshold_condition == False], y_subset[threshold_condition == False]
        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """
        y_left = y_subset[X_subset[:, feature_index] <= threshold]
        y_right = y_subset[X_subset[:, feature_index] > threshold]
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset, parents_path, depth):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        parents_path : list of strings with length not bigger than self.last_n_features
            Names of last self.last_n_features decision features in path to root

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        n_objects, n_features = X_subset.shape
        feature_index = 0
        threshold = 0
        G = np.inf
        H = self.criterion

        for feature_idx in range(n_features):
            feature_name = self.features_names[feature_idx]

            if (feature_name in parents_path) or (depth in self.features_in_levels.keys() and feature_name in self.features_in_levels[depth]):
                continue

            thresholds = np.sort(np.unique(X_subset[:, feature_idx]))
            for t in thresholds:
                y_left, y_right = self.make_split_only_y(feature_idx, t, X_subset, y_subset)
                G_new = (len(y_left) * H(y_left) + len(y_right) * H(y_right)) / n_objects
                if len(y_left) < len(y_right):
                    G_new += self.c * (1 - len(y_left) / (len(y_right) + 1))
                else:
                    G_new += self.c * (1 - len(y_right) / (len(y_left) + 1))
                if isnan(G_new):
                    continue
                if G_new < G:
                    threshold = t
                    feature_index = feature_idx
                    G = G_new

        return feature_index, threshold, G

    def set_leaf(self, leaf_id, y_subset):
        self.children_left[leaf_id] = -1
        self.children_right[leaf_id] = -1
        self.impurity[leaf_id] = -1
        self.feature[leaf_id] = -1
        self.threshold[leaf_id] = -1
        if len(y_subset) > 0:
            if self.classification:
                self.value[leaf_id] = np.mean(y_subset, axis=0)
                self.probas[leaf_id] = [1 - np.mean(y_subset, axis=0), np.mean(y_subset, axis=0)]
            elif self.criterion == 'variance':
                self.value[leaf_id] = np.mean(y_subset)
            else:
                self.value[leaf_id] = np.median(y_subset)

    def make_tree(self, X_subset, y_subset, parents_path=[]):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        feature_index, threshold, impurity = self.choose_best_split(X_subset, y_subset, parents_path, self.depth + 1)
        new_node = Node(feature_index, threshold, node_id=self.last_node_id, total_samples=self.n_samples, cur_samples=X_subset.shape[0])
        self.node_count += 1
        self.depth += 1
        self.last_node_id += 1
        self.n_node_samples[new_node.node_id] = X_subset.shape[0]
        feature_name = self.features_names[feature_index]
        if self.use_layer_constraint:
            if self.depth not in self.features_in_levels.keys():
                self.features_in_levels[self.depth] = [feature_name]
            else:
                self.features_in_levels[self.depth].append(feature_name)

        parents_path.append(feature_name)
        if len(parents_path) >= self.last_n_features_size + 1:
            parents_path = parents_path[-self.last_n_features_size:]

        if self.depth < self.max_depth and X_subset.shape[0] >= self.min_samples_split:
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)

            if min(len(y_left), len(y_right)) > 0:
                new_node.left_child = self.make_tree(X_left, y_left, parents_path.copy())
                new_node.right_child = self.make_tree(X_right, y_right, parents_path.copy())

                self.children_left[new_node.node_id] = new_node.left_child.node_id
                self.children_right[new_node.node_id] = new_node.right_child.node_id
                if self.classification:
                    self.value[new_node.node_id] = np.mean(y_subset, axis=0)
                    self.probas[new_node.node_id] = [1 - np.mean(y_subset, axis=0), np.mean(y_subset, axis=0)]
                else:
                    self.value[new_node.node_id] = np.mean(y_subset)

                self.impurity[new_node.node_id] = impurity
                self.feature[new_node.node_id] = feature_index
                self.threshold[new_node.node_id] = threshold
            else:
                self.set_leaf(new_node.node_id, y_subset)
        else:
            self.set_leaf(new_node.node_id, y_subset)

        self.depth -= 1
        
        self.max_depth = max(self.depth, self.max_depth)
        return new_node

    def dict_to_array(self, d, extra_val=-np.inf, type=int):
        max_idx = np.max(list(d.keys())) + 1
        a = np.zeros((max_idx), dtype=type)
        for i in range(max_idx):
            if i in d.keys():
                if abs(d[i]) < 1e9: 
                    a[i] = d[i]
                else:
                    a[i] = int(1e9)
            else:
                a[i] = extra_val
        return a

    def fit(self, X, y):
        self.n_samples, self.n_features_ = X.shape
        self.tree_ = self.make_tree(X, y)
        if self.classification:
            self.tree_.n_classes = {0: 1, 1: len(np.unique(y))}
            self.classes_ = y
        else:
            self.tree_.n_classes = {0: 1, 1: 1}
        self.is_fitted_ = True
        self.tree_.impurity = self.dict_to_array(self.impurity)
        self.tree_.children_left = self.dict_to_array(self.children_left)
        self.tree_.children_right = self.dict_to_array(self.children_right)
        self.tree_.value = self.dict_to_array(self.value, type=np.float)
        
        self.tree_.impurity = self.dict_to_array(self.impurity, type=np.float)
        self.tree_.feature = self.dict_to_array(self.feature)
        self.tree_.threshold = self.dict_to_array(self.threshold, type=np.float)
        self.tree_.n_node_samples = self.dict_to_array(self.n_node_samples)
        self.tree_.node_count = self.node_count

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        n_objects = X.shape[0]
        y_predicted = np.zeros(n_objects)
        self.recursive_predict(0, X, np.arange(n_objects), y_predicted)

        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'
        n_objects = X.shape[0]
        y_predicted_probs = np.zeros((n_objects, self.tree_.n_classes[1]))
        self.recursive_predict(0, X, np.arange(n_objects), y_predicted_probs, probas=True)
        return y_predicted_probs

    def recursive_predict(self, node_id, X, indices, y_predicted, probas=False):
        if self.children_left[node_id] == -1:
            if probas:
                y_predicted[indices] = self.value[node_id]
            else:
                y_predicted[indices] = self.probas[node_id]
        else:
            (X_left, left_indices), (X_right, right_indices) = self.make_split(self.feature[node_id], self.value[node_id], X,
                                                                               indices)
            self.recursive_predict(self.children_left[node_id], X_left, left_indices, y_predicted)
            self.recursive_predict(self.children_right[node_id], X_right, right_indices, y_predicted)
