from typing import List


class Node:
    def __init__(self, left: 'Node' = None, right: 'Node' = None, parent: 'Node' = None, gini: float = 0, feature_index: int = 0, threshold: float = 0, class_label: int = -1):
        self.left: 'Node' = left
        self.right: 'Node' = right
        self.parent: 'Node' = parent
        self.gini: float = gini
        self.feature_index: int = feature_index
        self.threshold: float = threshold
        self.class_label: int = class_label


class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        self.max_depth: int = max_depth
        self.root: Node = None
        self.X: List[List[float]] = None
        self.y: List[int] = None
        self.min_y: int = 0
        self.max_y: int = 0

    def fit(self, X: List[List[float]], y: List[int]):
        self.X = X
        self.y = y
        self.min_y = min(self.y)
        self.max_y = max(self.y)
        self.root = Node()
        self.build_tree(self.root, 0, len(self.y) - 1)

    def predict(self, X: List[List[float]]) -> List[int]:
        res: List[int] = []
        for i in range(len(X)):
            row = X[i]
            traverse: Node = self.root
            while True:
                if traverse.class_label != -1:
                    res.append(traverse.class_label)
                    break
                feature = traverse.feature_index
                value = row[feature]
                if value <= traverse.threshold:
                    traverse = traverse.left
                else:
                    traverse = traverse.right
        return res

    def build_tree(self, node: Node, depth: int, end_index: int) -> None:
        feature_list: List = [column for column in range(len(self.X[0]))]
        if node.parent is not None:
            feature_list.remove(node.parent.feature_index)

        if depth == self.max_depth or self.is_all_same(node, end_index):
            self.decide_class_label(node, end_index)
            return

        self.best_split(node, feature_list, end_index)

        node.left = Node(parent=node)
        node.right = Node(parent=node)
        left_end_index = self.arrange_for_left(node, end_index)
        self.build_tree(node.left, depth + 1, left_end_index)
        right_end_index = self.arrange_for_right(node, end_index)
        self.build_tree(node.right, depth + 1, right_end_index)

    def best_split(self, node: Node, feature_list: List, end_index: int) -> None:
        gini_lists: List[List[float]] = [
            [2 for y in range(len(self.X[0]))] for x in range(end_index)]
        consecutive_averages: List[List[float]] = [[-1 for y in range(len(self.X[0]))]
                                                   for x in range(end_index)]
        for j in feature_list:
            self.sort_by_column(j, end_index)
            for i in range(end_index):
                element1 = self.X[i][j]
                element2 = self.X[i + 1][j]
                avg = (element1 + element2) / 2
                consecutive_averages[i][j] = avg

        for j in feature_list:
            self.sort_by_column(j, end_index)
            for i in range(len(consecutive_averages)):
                element = consecutive_averages[i][j]
                self.calculate_gini_impurity(
                    element, j, gini_lists[i], end_index)

        min_ginis: List[float] = [2 for x in range(len(self.X[0]))]
        min_gini_row_indices: List[int] = [-1 for x in range(len(self.X[0]))]
        for j in feature_list:
            min_gini = gini_lists[0][j]
            min_gini_index = 0
            for i in range(len(gini_lists)):
                gini = gini_lists[i][j]
                if gini < min_gini:
                    min_gini = gini
                    min_gini_index = i
            min_ginis[j] = min_gini
            min_gini_row_indices[j] = min_gini_index

        min_gini = min_ginis[0]
        min_gini_index = 0
        min_gini_row_index = 0
        for i in range(len(min_ginis)):
            if min_ginis[i] < min_gini:
                min_gini = min_ginis[i]
                min_gini_index = i
                min_gini_row_index = min_gini_row_indices[i]
        node.feature_index = min_gini_index
        node.threshold = consecutive_averages[min_gini_row_index][min_gini_index]
        node.gini = min_gini

    def arrange_for_left(self, node: Node, end_index: int) -> int:
        last_swapped_index = 0
        for i in range(end_index + 1):
            element = self.X[i][node.feature_index]
            if element <= node.threshold:
                self.X[i], self.X[last_swapped_index] = self.X[last_swapped_index], self.X[i]
                self.y[i], self.y[last_swapped_index] = self.y[last_swapped_index], self.y[i]
                last_swapped_index += 1
        return last_swapped_index - 1

    def arrange_for_right(self, node: Node, end_index: int) -> int:
        last_swapped_index = 0
        for i in range(end_index + 1):
            element = self.X[i][node.feature_index]
            if element > node.threshold:
                self.X[i], self.X[last_swapped_index] = self.X[last_swapped_index], self.X[i]
                self.y[i], self.y[last_swapped_index] = self.y[last_swapped_index], self.y[i]
                last_swapped_index += 1
        return last_swapped_index - 1

    def calculate_gini_impurity(self, average: float, feature_index: int, ginis: List[float], end_index: int) -> None:
        left_values = [0 for x in range(self.min_y, self.max_y + 1, 1)]
        right_values = [0 for x in range(self.min_y, self.max_y + 1, 1)]
        for i in range(end_index + 1):
            label = self.y[i]
            if self.X[i][feature_index] < average:
                left_values[label] += 1
            else:
                right_values[label] += 1
        p_k = 0
        left_sum = sum(left_values)
        if left_sum != 0:
            for i in range(len(left_values)):
                prob = left_values[i] / left_sum
                p_k += prob**2
        gini1 = 1 - p_k
        p_k = 0
        right_sum = sum(right_values)
        if right_sum != 0:
            for i in range(len(right_values)):
                prob = right_values[i] / right_sum
                p_k += prob**2
        gini2 = 1 - p_k
        all_sum = left_sum + right_sum
        weighted_gini = gini1 * left_sum / all_sum + gini2 * right_sum / all_sum
        ginis[feature_index] = weighted_gini

    def decide_class_label(self, node: Node, end_index: int) -> None:
        counts: List[int] = [0 for x in range(self.min_y, self.max_y + 1, 1)]
        for i in range(end_index + 1):
            counts[self.y[i]] += 1
        max_index = 0
        max_value = counts[0]
        for i in range(len(counts)):
            if counts[i] > max_value:
                max_value = counts[i]
                max_index = i
        node.class_label = max_index

    def is_all_same(self, node: Node, end_index: int) -> bool:
        same: bool = True
        for i in range(end_index):
            if self.y[i] != self.y[i + 1]:
                same = False
        return same

    def sort_by_column(self, feature_index: int, end_index: int) -> None:
        i = 0
        while i < end_index:
            val1 = self.X[i][feature_index]
            val2 = self.X[i + 1][feature_index]
            if val1 > val2:
                self.X[i], self.X[i + 1] = self.X[i + 1], self.X[i]
                self.y[i], self.y[i + 1] = self.y[i + 1], self.y[i]
                if i == 0:
                    i = -1
                else:
                    i -= 2
            i += 1

    def print_tree(self, node: Node, status: str) -> None:
        print('feature_index: ' + str(node.feature_index) + '\nthreshold: ' +
              str(node.threshold) + '\nclass_label: ' + str(node.class_label) + '\nstatus: ' + status + '\n')
        if node.left != None:
            self.print_tree(node.left, 'left')
        if node.right != None:
            self.print_tree(node.right, 'right')