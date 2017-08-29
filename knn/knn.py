import sys
import pickle
import numpy as np
from sklearn.decomposition import PCA


def calculate_distance(a, b):
    distance = np.sqrt(np.sum((a - b) ** 2))
    return distance


def find_class(neighbors):
    vote_dict = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        distance = neighbor[1]
        if label in vote_dict:
            # voting weight using inverse euclidean distance
            vote_dict[label] += 1 / distance
        else:
            vote_dict[label] = 1 / distance
    return max(vote_dict.items(), key=lambda x: x[1])[0]


def find_neighbors(k, train_pca, test_pca, train_labels):
    distance = []
    neighbors = []
    for i in range(len(train_pca)):
        distance.append((train_pca[i], calculate_distance(train_pca[i], test_pca), train_labels[i]))
    distance.sort(key=lambda x: x[1])
    for x in range(k):
        neighbors.append(distance[x])
    return neighbors


def convert_gray(data, row, column):
    img = data[:]
    grayed = np.zeros((row, column))
    for i in range(len(img)):
        grayed[i] = img[i][:1024] * 0.299 + img[i][1024:2048] * 0.587 + img[i][2048:] * 0.114
    return grayed


def do_pca(gray_data, d):
    pca = PCA(n_components=d, svd_solver='full')
    pca_obj = pca.fit(gray_data)
    train_pca = pca_obj.transform(gray_data)
    # print(train_pca.shape)
    return pca_obj, train_pca


def reduce_dimension(gray_data, pca):
    return pca.transform(gray_data)


def write_output(file, result):
    with open(file, 'w') as f:
        for res in result:
            f.write('{0} {1}\n'.format(res[0], res[1]))
    f.close()


def knn(train_data, test_data, test_labels, train_labels, k, d):
    # do pca and reduce dimension for train data
    pca_obj, train_pca = do_pca(train_data, d)
    # reduce dimension for test data
    test_pca = reduce_dimension(test_data, pca_obj)
    # perform kNN for each test sample
    labels = []
    for i in range(len(test_pca)):
        neighbors = find_neighbors(k, train_pca, test_pca[i], train_labels)
        prediction = find_class(neighbors)
        actual = test_labels[i]
        labels.append((prediction, actual))
    return labels


def main():
    if len(sys.argv) == 5:
        k = int(sys.argv[1])  # number of nearest neighbor
        d = int(sys.argv[2])  # number of pca dimension
        n = int(sys.argv[3])  # number of test samples to consider
        input_data = sys.argv[4]
        total_samples = 1000  # total number of train+test samples

        # load image data
        with open(input_data, 'rb') as fo:
            image_data = pickle.load(fo, encoding='bytes')
        sampled_data = image_data[b'data'][:total_samples]
        sampled_labels = image_data[b'labels'][:total_samples]

        # create train and test splits
        train_data = sampled_data[n:]
        test_data = sampled_data[:n]
        test_labels = sampled_labels[:n]
        train_labels = sampled_labels[n:]

        # convert the image to gray scale
        train_grayed = convert_gray(train_data, 1000 - n, 1024)
        test_grayed = convert_gray(test_data, n, 1024)

        # perform kNN using PCA
        labels = knn(train_grayed, test_grayed, test_labels, train_labels, k, d)

        # output the predicted and actual labels for every image
        write_output('output.txt', labels)


if __name__ == '__main__':
    main()
