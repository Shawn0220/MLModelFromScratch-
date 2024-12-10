class Node:
  def __init__(self, attribute=None, leaf=False, classification=None):
    self.label = None
    self.children = {} #  字典，存储子节点，键为属性值，值为子节点
    # extra members
    self.leaf = leaf  # 是否为叶节点
    self.attribute = attribute  # 当前节点划分的属性, None when its leaf
    self.classification = classification  # not None when its leaf 如果是叶节点，则存储分类结果
    self.father_certain_class = [] # remember entropy decay, may works in pruning
  def printer(self):
    print(f"Label: {self.label}")
    print(f"Children: {self.children}")
    print(f"Attribute: {self.attribute}")
    print(f"Leaf: {self.leaf}")
    print(f"Classification: {self.classification}")