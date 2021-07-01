import random

class Node():
    def __init__(self, left, right, is_leaf = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.idx = idx

        if not self.is_leaf:
            self.value = left.value + right.value

        self.parent = None

        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, True, idx)
        leaf.value = value
        return leaf


class SumTree():
    def __init__(self, weights: list):
        nodes = [Node.create_leaf(v, i) for i,v in enumerate(weights)]
        self.leaf_nodes = nodes
        while len(nodes) > 1:
            inodes = iter(nodes)
            nodes = [Node(*pair) for pair in zip(inodes, inodes)]

        self.top_node = nodes[0]
        
    def retrieve(self, value: float, node: Node):
        if node.is_leaf:
            return node
        if node.left.value >= value:
            return self.retrieve(value, node.left)
        else:
            return self.retrieve(value - node.left.value, node.right)

    def update(self, idx, new_value: float):
        node = self.leaf_nodes[idx]
        change = new_value - node.value
        node.value = new_value
        self.propagate_changes(change, node.parent)

    def propagate_changes(self, change:float, node: Node):
        node.value += change
        if node.parent is not None:
            self.propagate_changes(change, node.parent)

    def draw_idx(self):
        u = random.uniform(0, self.top_node.value)
        return self.retrieve(u, self.top_node).idx
