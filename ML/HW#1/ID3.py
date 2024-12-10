from node import Node
import math
from collections import Counter
import random
from parse import parse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  # filling missing labels with mode
  examples = fill_missing_with_mode(examples)
  target_attribute = 'Class'
  # Define: most_common - positive / other-wise negative
  # no sample, return defualt 
  if len(examples) == 0:
      return Node(leaf=True, classification=default)

  
  class_labels = [example[target_attribute] for example in examples]
  class_counts = Counter(class_labels)
  most_common_class_cnt, most_common_class = 0, ''
  for k, v in class_counts.items():
    if most_common_class_cnt<v:
      most_common_class_cnt = v
      most_common_class = k
  # print("class_counts", most_common_class)
  # print(class_counts)
  for k in class_counts.keys():
    if k!=most_common_class:
      negative_class = k
  
  
  # pos
  at_least_one_not_common = False
  for exa in examples:
    if exa[target_attribute] != most_common_class:
      at_least_one_not_common = True
      break
  if not at_least_one_not_common:
     return Node(leaf=True, classification=most_common_class)
  # neg
  at_least_one_not_neg = False
  for exa in examples:
     if exa[target_attribute] != negative_class:
        at_least_one_not_neg = True
        break
  if not at_least_one_not_neg:
     return Node(leaf=True, classification=negative_class)

  # if all(example['Class'] == most_common_class for example in examples):
  #     return Node(is_leaf=True, classification=most_common_class)
  
  
  attributes = examples[0].keys() - {target_attribute}
  if len(attributes) == 0:
      print("return Node(leaf=True, classification=most_common_class)", most_common_class)
      return Node(leaf=True, classification=most_common_class)
  
  
  max_info = -1
  best_attribute = ''
  candidate_atts = list(examples[0].keys() - {target_attribute})
  for cand in candidate_atts:
    # print("split by ", cand)
    info_gained_by_cand = cal_info_gained(examples, cand, target_attribute)
    # print("info gained", cal_info_gained(examples, cand, target_attribute))
    if info_gained_by_cand > max_info:
       best_attribute = cand
       max_info = info_gained_by_cand
     
  # best_attribute = max(attributes, key=lambda attr: cal_info_gained(examples, attr))
  
  
  root = Node(attribute=best_attribute)
  
  
  att_values = set(example.get(best_attribute, '?') for example in examples)
  # print(att_values)
  
  for value in att_values:
      
      subset = [example for example in examples if example.get(best_attribute, '?') == value]
      subset_without_key = [{k: v for k, v in sub.items() if k != best_attribute} for sub in subset]
    
      # Recursively build the subtree
      subtree = ID3(subset_without_key, most_common_class)
    
      # Connect the subtree to the current node
      root.children[value] = subtree
     
  
  return root


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  # Base case: If the node is a leaf, there's nothing to prune
  if node.leaf:
      return

  class_of_node = get_leaf_classifications(node)
  son_certain_class = get_node_leaf_classifications(node)

  # print(certain_class_of_node)
  
  filtered_class_of_node = [x for x in class_of_node if x not in node.father_certain_class]

  # if filtered_a is not null，randomly select a number from filtered_a
  if filtered_class_of_node:
      count = Counter(filtered_class_of_node)
      when_no_true_on_val = count.most_common(1)[0][0]
    #   when_no_true_on_val = random.choice(filtered_class_of_node)
  else:
      when_no_true_on_val = class_of_node[0]
      

  # Recursively prune each child of the current node
  for value, child in node.children.items():
      child.father_certain_class = son_certain_class
      prune(child, examples)

  # After pruning children, try converting this node to a leaf
  original_accuracy = test(node, examples)

  # Save the original attribute and children for restoration if pruning fails
  original_attribute = node.attribute
  original_children = node.children

  # Calculate the majority class among the examples that pass through this node
  class_counts = {}


  for example in examples:
      # print("printing node-------------------------------")
      
      if evaluate(node, example) == example['Class']:
          label = example['Class']
          class_counts[label] = class_counts.get(label, 0) + 1
      # Handle the case where class_counts is empty
  if class_counts:
      majority_class = max(class_counts, key=class_counts.get)
  else:
      # If no examples pass through this node, fallback to a default class
      # For simplicity, use a default class such as 0 or 1
      majority_class = when_no_true_on_val

  # Convert the node into a leaf node
  node.leaf = True
  node.classification = majority_class
  node.attribute = None
  node.children = {}

  # Check the accuracy after pruning
  pruned_accuracy = test(node, examples)
  # If pruning reduces accuracy, restore the original structure
  if pruned_accuracy < original_accuracy:
      node.leaf = False
      node.classification = None
      node.attribute = original_attribute
      node.children = original_children



def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  correct = 0

  for example in examples:
      prediction = classify_example(node, example)  
      if prediction == example['Class']:  
          correct += 1

  accuracy = correct / len(examples)
  return accuracy

def classify_example(node: Node, example: dict) -> str:
    # print(f"Current node attribute: {node.attribute}, Example: {example}")  
    if node.leaf:
        return node.classification

    attribute_value = example.get(node.attribute)
    if attribute_value in node.children:
        return classify_example(node.children[attribute_value], example)
    else:
        return node.classification  # If no matching child, return the majority class at this node


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  # if leaf, return class
  if node.leaf:
      return node.classification

  
  attribute_value = example.get(node.attribute)

  
  child_node = node.children.get(attribute_value)

  
  if child_node is None:
      return None 

  
  return evaluate(child_node, example)


# calculate entropy
def cal_entropy(samples, target_attribute):
  # Count the frequency of each class of target_attribute
  counter = Counter([row[target_attribute] for row in samples])
  # print(counter)
  total_samples = len(samples)
  # Calculate the entropy using the formula
  ent = -sum((count/total_samples) * math.log2(count/total_samples) for count in counter.values() if count != 0)
  return ent


def cal_info_gained(samples, split_attribute, target_attribute):
  # print("samples: ", samples)
  total_entropy = cal_entropy(samples, target_attribute)
  split_by_att = {}
  for samp in samples:
      # get split_att's class {1:[] ,2:[] ,3:[]}
      attribute_class = samp[split_attribute]
      if attribute_class not in split_by_att:
          split_by_att[attribute_class] = [samp]
      else:
          split_by_att[attribute_class].append(samp)
  
  sum_entropy = 0
  for sp in split_by_att.values():
      sum_entropy += cal_entropy(sp, target_attribute) * (len(sp)/len(samples))
  # print("sum_entropy ", sum_entropy)
  
  # Information gain is the difference between the entropy before and after the split
  info = total_entropy - sum_entropy
  return info


def get_att_mode(values):
  
  filtered_values = [v for v in values if v != '?']
  if not filtered_values:
      return None
  
  count = Counter(filtered_values)
  max_freq = max(count.values())
  
  modes = [k for k, v in count.items() if v == max_freq]
  # randomly select a mode
  return random.choice(modes)


def fill_missing_with_mode(data):
  
  keys = data[0].keys()
  
  for key in keys:
      
      column_values = [row[key] for row in data]
      # compute mode
      mode_value = get_att_mode(column_values)
      
      for row in data:
          if row[key] == '?':
              row[key] = mode_value
  return data


def print_tree(node, depth=0):
    # Print indentation based on the depth of the node
    indent = "  " * depth
    
    if node.leaf:
        print(f"{indent}Leaf: Class = {node.classification}")
    else:
        print(f"{indent}Node: Split by '{node.attribute}'")
        # Recursively print the children nodes
        for value, child in node.children.items():
            print(f"{indent}  Branch: {node.attribute} = {value}")
            print_tree(child, depth + 1)

def get_leaf_classifications(node):
    leaf_classifications = []
    
   
    if node.leaf:
        return [node.classification]
    
   
    for child in node.children.values():
        leaf_classifications.extend(get_leaf_classifications(child))
    
    return leaf_classifications

def get_node_leaf_classifications(node):
    nearest_leaf_classifications = []

    for id, child in node.children.items():
      if child.leaf:
         nearest_leaf_classifications.append(child.classification)
  
    return nearest_leaf_classifications


# ID3 with features selections for Random Forest
def ID3_with_features(examples, default, features):
 
    examples = fill_missing_with_mode(examples)
    target_attribute = 'Class'
    
    
    if len(examples) == 0:
        return Node(leaf=True, classification=default)
    
   
    class_labels = [example[target_attribute] for example in examples]
    if len(set(class_labels)) == 1:
        return Node(leaf=True, classification=class_labels[0])
    
    
    if len(features) == 0:
        most_common_class = Counter(class_labels).most_common(1)[0][0]
        return Node(leaf=True, classification=most_common_class)
    
   
    best_attribute = max(features, key=lambda attr: cal_info_gained(examples, attr, target_attribute))
    
   
    root = Node(attribute=best_attribute)
    
  
    att_values = set(example[best_attribute] for example in examples)
    for value in att_values:
        subset = [example for example in examples if example[best_attribute] == value]
        subset_without_key = [{k: v for k, v in sub.items() if k != best_attribute} for sub in subset]
        subtree = ID3_with_features(subset_without_key, default, features - {best_attribute})
        root.children[value] = subtree
    
    return root

class RandomForest:
    def __init__(self, num_trees, max_features=None):
       
        self.num_trees = num_trees
        self.max_features = max_features
        self.trees = []
        self.feature_subsets = []  # 
    def bootstrap_sample(self, examples):
        
        n = len(examples)
        return [random.choice(examples) for _ in range(n)]
    
    def select_features(self, examples):
        
        all_features = list(examples[0].keys() - {'Class'})  
        if self.max_features is None or self.max_features > len(all_features):
            return all_features
        return random.sample(all_features, self.max_features)
    
    def fit(self, examples, default):
     
        for _ in range(self.num_trees):
            
            sample = self.bootstrap_sample(examples)
            
            
            selected_features = self.select_features(sample)
            self.feature_subsets.append(selected_features)
            
            
            tree = ID3_with_features(sample, default, set(selected_features))
            self.trees.append(tree)
    
    def predict(self, example):
        
        predictions = [
            classify_example(tree, {k: v for k, v in example.items() if k in features})
            for tree, features in zip(self.trees, self.feature_subsets)
        ]
       
        majority_vote = Counter(predictions).most_common(1)[0][0]
        return majority_vote
    
    def score(self, examples):
        
        correct = sum(self.predict(example) == example['Class'] for example in examples)
        return correct / len(examples)



def train_random_forest_and_evaluate(num_trees_range, train_data, test_data):
    train_accuracies = []
    test_accuracies = []

    for num_trees in num_trees_range:
        
        rf = RandomForest(num_trees=num_trees, max_features=int(len(train_data[0].keys())**0.5))
        rf.fit(train_data, default=0)  
        train_accuracy = rf.score(train_data)
        test_accuracy = rf.score(test_data)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    return train_accuracies, test_accuracies



data = parse('candy.data')  
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)


num_trees_range = range(2, 51)
train_accuracies, test_accuracies = train_random_forest_and_evaluate(num_trees_range, train_data, test_data)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(num_trees_range, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(num_trees_range, test_accuracies, label='Test Accuracy', marker='x')
plt.title('Random Forest Accuracy vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()







      