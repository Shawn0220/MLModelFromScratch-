import numpy as np
from collections import Counter, defaultdict
import math
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import homogeneity_score
# returns Euclidean distance between vectors and b
def euclidean(a,b):
    return math.sqrt(sum((float(a_v) - float(b_v)) ** 2 for a_v, b_v in zip(a, b)))

def cos_dis(a,b):
    cos_sim = cosim(a,b)
    return 1-cos_sim

# returns Cosine Similarity between vectors and b
def cosim(a,b):
    dot_product = sum(float(x) * float(y) for x, y in zip(a, b))
    magnitude1 = math.sqrt(sum(float(x) ** 2 for x in a))
    magnitude2 = math.sqrt(sum(float(y) ** 2 for y in b))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def pearson_correlation(a, b):
    """
    Calculate the Pearson Correlation Coefficient between two vectors a and b.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("Vectors must be of the same length")
    
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    
    numerator = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    denominator = math.sqrt(sum((a[i] - mean_a) ** 2 for i in range(n)) * sum((b[i] - mean_b) ** 2 for i in range(n)))
    
    return numerator / denominator if denominator != 0 else 0

def hamming_distance(a, b):
    """
    Calculate the Hamming Distance between two vectors a and b.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length")
    
    return sum(1 for a_v, b_v in zip(a, b) if a_v != b_v)

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
   
    predictions = []
    k = 5
    for query_label, query_pixel in query:
        # cal distance
        distances = []
        for train_label, train_pixel in train:
            if metric == "euclidean":
                dist = euclidean(train_pixel, query_pixel)
            elif metric == "cosim":
                dist = cosim(train_pixel, query_pixel)
                dist = -dist
            distances.append((dist, train_label))

        # find k nearest neibours
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]

        # votes
        k_labels = [label for _, label in k_nearest]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)

    return predictions

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    #cosim
    k=100
    max_iter = 5
    train_data = np.array([np.array(sample[1], dtype=float) for sample in train])

    np.random.seed(42)
    centroids = train_data[np.random.choice(len(train_data), k, replace=False)]
    # print("cents:\n", len(centroids))

    if metric == 'euclidean':
        distance_func = euclidean
    elif metric == 'cosim':
        distance_func = cos_dis

    for _ in range(max_iter):
        clusters = defaultdict(list)
        for i, sample in enumerate(train_data):
            distances = [distance_func(sample, centroid) for centroid in centroids]
            # print(distances)
            closest_centroid = np.argmin(distances)
            # print("closest_centroid", closest_centroid)
            clusters[closest_centroid].append(i)
            # print(clusters)

   
        new_centroids = []
        for i in range(k):
            if clusters[i]:  
             
                elements_in_cluster = train_data[clusters[i]]
                new_centroid = np.mean(elements_in_cluster, axis=0)
            else:
              
                new_centroid = centroids[i]

            new_centroids.append(new_centroid)
        # print(new_centroids)

        # check if anything change
        new_centroids = np.array(new_centroids)
        if np.all(centroids == new_centroids):
            break  
        centroids = new_centroids

    
    cluster_labels = {}
    # print("clusters:", clusters)
    for cluster_id, indices in clusters.items():
        labels = [train[i][0] for i in indices]  
        if labels:
            most_common_label = Counter(labels).most_common(1)[0][0]
            cluster_labels[cluster_id] = most_common_label
        else:
            cluster_labels[cluster_id] = -1

    
    query_data = np.array([np.array(sample[1], dtype=float) for sample in query])
    predicted_labels = []
    for sample in query_data:
        distances = [distance_func(sample, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        predicted_labels.append(cluster_labels[closest_centroid])

    return predicted_labels


def normalize_data(data):
 
    normalized_data = []
    for label, pixels in data:
        normalized_pixels = [int(p) / 255.0 for p in pixels]
        normalized_data.append([label, normalized_pixels])
    return normalized_data
    
def binarize_data(data, threshold=60):
   
    binary_data = []
    for label, pixels in data:
        binary_pixels = [1 if int(p) >= threshold else 0 for p in pixels]
        binary_data.append([label, binary_pixels])
    return binary_data

def extract_labels(data):
    return [label for label, _ in data]

def calculate_accuracy(ground_truth, predictions):
    correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
    accuracy = correct / len(ground_truth)
    return accuracy

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    # print(data_set[0])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('mnist_valid.csv','pixels')
    
if __name__ == "__main__":
    #Task1 KNN
    train_data = read_data("mnist_train.csv")
    test_data = read_data("mnist_test.csv")

    # print("Using Normalization")
    # train_data = normalize_data(train_data)
    # test_data = normalize_data(test_data)
    print("Using Binarization")
    train_data = binarize_data(train_data)
    test_data = binarize_data(test_data)    

    # GT lables
    ground_truth = extract_labels(test_data)
    # Euclidean Distance
    start_time = time.time()  
    predictions = knn(train_data, test_data, metric="euclidean")
    end_time = time.time()  

    print("_____Using euclidean distance_____")
    print(f"euclidean program execution time: {end_time - start_time} s")
    # print("pred:\n", predictions)
   
    conf_matrix = confusion_matrix(ground_truth, predictions)
    
    accuracy = calculate_accuracy(ground_truth, predictions)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Cosim Similarity
    start_time = time.time()  
    predictions = knn(train_data, test_data, metric="cosim")
    end_time = time.time()  
    print("_____Using cosim similarity_____")
    print(f"cosim program execution time: {end_time - start_time} s")
    
    conf_matrix = confusion_matrix(ground_truth, predictions)
  
    accuracy = calculate_accuracy(ground_truth, predictions)

   
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    '''
    binarize data | threshold=90

    euclidean program execution time: 32.89244556427002 s
    Accuracy: 89.50%

    threshold=200 79% 91%
    threshold=160 89% 90.5%
    threshold=128 89% 92%
    threshold=90 89.5% 93%
    threshold=60 91.5% 93%
    threshold=30 90.5% 92%

    norm 89.5% 92%
    '''
   
    train_data = read_data('mnist_train.csv')
    test_data = read_data('mnist_test.csv')

    ground_truth = extract_labels(test_data)

    # train_data = normalize_data(train_data)
    # test_data = normalize_data(test_data)

   
    print("___Cosim Similarity___")
    predicted_labels = kmeans(train_data, test_data, metric='cosim')

    #print(predicted_labels)
    
    
    conf_matrix = confusion_matrix(ground_truth, predicted_labels)
 
    accuracy = calculate_accuracy(ground_truth, predicted_labels)


    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    #homogeneity score
    homogeneity = homogeneity_score(ground_truth, predicted_labels)
    print("Homogeneity Score:", homogeneity)

    print("___Euclidean___")
    predicted_labels = kmeans(train_data, test_data, metric='euclidean')
    
    conf_matrix = confusion_matrix(ground_truth, predicted_labels)
 
    accuracy = calculate_accuracy(ground_truth, predicted_labels)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    #homogeneity score
    homogeneity = homogeneity_score(ground_truth, predicted_labels)
    print("Homogeneity Score:", homogeneity)