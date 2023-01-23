import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain = 0, count = 0, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
        self.count = count

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100,  depth=None, n_features=None, criterion = "GINI"):
        self.labels = None
        self.min_samples_splits = min_samples_split
        self.max_depth = max_depth
        self.n_features= n_features
        self.root = None
        self.depth = depth if depth else 0
        self.criterion = criterion

    def fit(self, X, y):
        # Number of features do not exceed the actua feature we have
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.labels = X.columns
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        count = Counter(y)
        # check the stopping criteria
        if depth>=self.max_depth or n_labels == 1 or n_samples<self.min_samples_splits:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        # find the best split
        best_feature, best_thresh, gain = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X.iloc[:, best_feature], best_thresh)
        left = self._grow_tree(X.iloc[left_idxs, :], y.iloc[left_idxs], depth + 1)
        right = self._grow_tree(X.iloc[right_idxs, :], y.iloc[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right, gain, count)

    def _best_split(self, X, y, feat_idxs):
        # initialize best_gain and gini_index
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X.iloc[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                if self.criterion == "GINI":
                    # calculate the GINI gain
                    gain = self.GINI_gain(y, X_column, thr)
                else:
                    # calculate the information gain
                    gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold, best_gain

    def GINI_gain(self, y, X_column, threshold):
        GINI_base = self.get_GINI(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        left_counts = Counter(y.iloc[left_idxs])
        right_counts = Counter(y.iloc[right_idxs])

        y0_left, y1_left, y0_right, y1_right = Counter(left_counts).get(0,0),Counter(left_counts).get(0,1), \
            Counter(right_counts).get(0,0), Counter(right_counts).get(0,1)

        gini_left = self.GINI_impurity(y0_left, y1_left)
        gini_right = self.GINI_impurity(y0_right, y1_right)

        # Getting the obs count from the left and the right data splits
        n_left = y0_left + y1_left
        n_right = y0_right + y1_right

        # Calculating the weights for each of the nodes
        w_left = n_left / (n_left + n_right)
        w_right = n_right / (n_left + n_right)

        #Calculate the GINI impurity
        wGINI = w_left * gini_left + w_right * gini_right

        #Calculate GINI gain
        GINI_gain = GINI_base - wGINI
        return GINI_gain

    def get_GINI(self, y):
        """
        Calculate the GINI impurity of a node
        """
        y1_count, y2_count = Counter(y).get(0, 0), Counter(y).get(1, 0)
        return self.GINI_impurity(y1_count, y2_count)

    def GINI_impurity(self, y1_count, y2_count):
        if y1_count is None:
            y1_count = 0
        if y2_count is None:
            y2_count = 0
        n = y1_count + y2_count
        # If n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return 0
        p1 = y1_count/n
        p2 = y2_count/n
        gini = 1 - (p1**2 + p2 **2 )
        return gini

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)
        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y.iloc[left_idxs]), self._entropy(y.iloc[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column.to_numpy() <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column.to_numpy() > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        counter = Counter(y)
        if len(Counter(y).most_common(1)) == 0:
            value = 0
        else:
            value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for idx, x in X.iterrows()])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold :
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_info(self, node, depth):
        """
        Method to print the information about the tree
        """
        if node.feature is None:
            return

        preamble0 = f"|{'---' * 3 * depth}"
        preamble1 = f"|{'   ' * 3 * depth}   |"

        print(f"{preamble0} Split rule: {self.labels[node.feature]} <= {node.threshold}")
        print(f"{preamble1} Gain info the node {round(node.gain, 5)}")
        print(f"{preamble1} Class distribution in the node {node.count}")
        # print(f"{preamble1} Predicted class {node.predict}")

        self.print_info(node.left, depth + 1)

        print(f"{preamble0} Split rule: {self.labels[node.feature]} > {node.threshold}")
        print(f"{preamble1} Gain info of the node {round(node.gain, 5)}")
        print(f"{preamble1} Class distribution in the node {node.count}")
        # print(f"{preamble1} Predicted class {node.predict}")

        self.print_info(node.right, depth + 1)

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        if self.root.is_leaf_node():
            return
        else:
            self.print_info(self.root, 1)
