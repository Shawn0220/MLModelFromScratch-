import ID3, parse, random

def testID3AndEvaluate():
  data = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0,c=0))
    if ans != 1:
      print("ID3 test failed.")
    else:
      print("ID3 test succeeded.")
  else:
    print("ID3 test failed -- no tree returned")

def testPruning(inFile):
  # data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  # validationData = [dict(a=0, b=0, c=1, Class=1)]
  data = [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0), dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0), dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0), dict(a=1, b=1, c=1, d=0, Class=0)]
  validationData = [dict(a=0, b=0, c=1, d=0, Class=1), dict(a=1, b=1, c=1, d=1, Class = 0)]
  tree = ID3.ID3(data, 0)
  print_tree(tree)
  ID3.prune(tree, validationData)
  print_tree(tree)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
    if ans != 1:
      print("pruning test failed.")
    else:
      print("pruning test succeeded.")
  else:
    print("pruning test failed -- no tree returned.")


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  ID3.print_tree(tree)
  fails = 0
  if tree != None:    
    acc = ID3.test(tree, trainData)
    print(acc)
    if acc == 1.0:
      print("testing on train data succeeded.")
    else:
      print("testing on train data failed.")
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print("testing on test data succeeded.")
    else:
      print("testing on test data failed.")
      fails = fails + 1
    if fails > 0:
      print("Failures: ", fails)
    else:
      print("testID3AndTest succeeded.")
  else:
    print("testID3andTest failed -- no tree returned.")	

def testTree(inFile,inFile1,inFile2):
  trainData = parse.parse(inFile)
  validData = parse.parse(inFile1)
  testData = parse.parse(inFile2)
  tree = ID3.ID3(trainData, 0)
  if tree != None:    
    print("Before Pruning:")
    acc = ID3.test(tree,trainData)
    print("cars_train acc:", acc)
    acc = ID3.test(tree,validData)
    print("cars_validation acc:",acc)
    acc = ID3.test(tree,testData)
    print("cars_test acc:",acc)
    ID3.prune(tree, validData)
    print("After Pruning:")
    acc = ID3.test(tree,trainData)
    print("cars_train acc:", acc)
    acc = ID3.test(tree,validData)
    print("cars_validation acc:",acc)
    acc = ID3.test(tree,testData)
    print("cars_test acc:",acc)

# def testpruning(inFile,inFile1,inFile2):
#   trainData = parse.parse(inFile)
#   validData = parse.parse(inFile1)
#   testData = parse.parse(inFile2)
#   tree = ID3.ID3(trainData, 0)
#   ID3.prune(tree,validData)
#   acc_0 = ID3.test(tree,trainData)
#   print("cars_train acc:", acc_0)
#   acc_1 = ID3.test(tree,validData)
#   print("cars_validation acc:",acc_1)
#   acc_2 = ID3.test(tree,testData)
#   print("cars_test acc:",acc_2)

# def testpruning(inFile,inFile1,inFile2):
#   trainData = parse.parse(inFile)
#   validData = parse.parse(inFile1)
#   testData = parse.parse(inFile2)
#   tree = ID3.ID3(trainData, 0)
#   acc = ID3.test(tree,trainData)
#   print("cars_train acc:", acc)
#   for i in range(50):
#     ID3.prune(tree, validData)
#     acc = ID3.test(tree,validData)
#     print("cars_validation acc:",acc)
#     acc = ID3.test(tree,testData)
#     print("cars_test acc:",acc)


def testpruning(inFile, inFile1, inFile2):
    trainData = parse.parse(inFile)
    validData = parse.parse(inFile1)
    testData = parse.parse(inFile2)
    
    # 构建决策树
    tree = ID3.ID3(trainData, 0)
    
    # 测试初始模型的性能
    acc_train = ID3.test(tree, trainData)
    print("Initial train acc:", acc_train)
    acc_valid = ID3.test(tree, validData)
    print("Initial validation acc:", acc_valid)
    acc_test = ID3.test(tree, testData)
    print("Initial test acc:", acc_test)
    
    # 通过validation数据集进行50轮剪枝
    for i in range(50):
        # 使用验证集进行剪枝
        ID3.prune(tree, validData)
        
        # 每次剪枝后，测试在验证集上的性能
        acc_valid = ID3.test(tree, validData)
        print(f"Epoch {i+1} validation acc:", acc_valid)
        
        # 测试在测试集上的性能
        acc_test = ID3.test(tree, testData)
        print(f"Epoch {i+1} test acc:", acc_test)


# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  withPruning = []
  withoutPruning = []
  data = parse.parse(inFile)
  # data_train = parse.parse(inFile)
  # data_valid = parse.parse(inFile1)
  # data_test = parse.parse(inFile2)
  for i in range(100):
    # train = data_train
    # valid = data_valid
    # test = data_test
    random.shuffle(data)
    train = data[:len(data)//2]
    valid = data[len(data)//2:3*len(data)//4]
    test = data[3*len(data)//4:]
  
    tree = ID3.ID3(train, 'democrat')
    acc = ID3.test(tree, train)
    print("training accuracy: ",acc)
    acc = ID3.test(tree, valid)
    print("validation accuracy: ",acc)
    acc = ID3.test(tree, test)
    print("test accuracy: ",acc)
  
    ID3.prune(tree, valid)
    acc = ID3.test(tree, train)
    print("pruned tree train accuracy: ",acc)
    acc = ID3.test(tree, valid)
    print("pruned tree validation accuracy: ",acc)
    acc = ID3.test(tree, test)
    print("pruned tree test accuracy: ",acc)
    withPruning.append(acc)
    tree = ID3.ID3(train+valid, 'democrat')
    acc = ID3.test(tree, test)
    print("no pruning test accuracy: ",acc)
    withoutPruning.append(acc)
  print(withPruning)
  print(withoutPruning)
  print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))

def print_tree(node, depth=0):
    """
    Recursively prints the decision tree structure.
    Each level of the tree is indented to show depth.
    """
    indent = "  " * depth  # Indentation for the current level

    if not node.children:  # If it's a leaf node
        print(f"{indent}Leaf: Predict {node.classification}")
    else:
        print(f"{indent}[{node.attribute}]")  # Print the splitting attribute
        for value, child in node.children.items():
            print(f"{indent}-- {node.attribute} = {value}:")
            print_tree(child, depth + 1)  # Recursively print the subtree



# testpruning("cars_train.data","cars_valid.data","cars_test.data")
# testpruning("cars_valid.data","cars_test.data")
#testpruning("cars_train.data","cars_valid.data","cars_test.data")
#testPruningOnHouseData("house_votes_84.data")
  
testTree("cars_train.data","cars_valid.data","cars_test.data")