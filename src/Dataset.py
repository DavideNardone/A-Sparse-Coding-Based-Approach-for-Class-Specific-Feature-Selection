from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import hdf5storage #dependency
import numpy as np
np.set_printoptions(threshold=np.inf)


class Dataset:
    def __init__(self, X, y):

        self.data = X
        self.target = y.flatten()

        # removing any row with at least one NaN value
        # TODO: remove also the corresponding target value
        self.data = self.data[~np.isnan(self.data).any(axis=1)]

        self.num_sample, self.num_features = self.data.shape[0], self.data.shape[1]

        # retrieving unique label for Dataset
        self.classes = np.unique(self.target)

    def standardizeDataset(self):

        # it simply standardize the data [mean 0 and std 1]
        if np.sum(np.std(self.data, axis=0)).astype('int32') == self.num_features and np.sum(
                np.mean(self.data, axis=0)) < 1 ** -7:
            print ('\tThe data were already standardized!')
        else:
            print ('Standardizing data....')
            self.data = StandardScaler().fit_transform(self.data)

    def normalizeDataset(self, norm):

        normalizer = preprocessing.Normalizer(norm=norm)
        self.data = normalizer.fit_transform(self.data)

    def scalingDataset(self):

        min_max_scaler = preprocessing.MinMaxScaler()
        self.data = min_max_scaler.fit_transform(self.data)

    def shufflingDataset(self):

        idx = np.random.permutation(self.data.shape[0])
        self.data = self.data[idx]
        self.target = self.target[idx]


    def split(self, split_ratio=0.8):

        # shuffling data
        indices = np.random.permutation(self.num_sample)

        start = int(split_ratio * self.num_sample)
        training_idx, test_idx = indices[:start], indices[start:]
        X_train, X_test = self.data[training_idx, :], self.data[test_idx, :]
        y_train, y_test = self.target[training_idx], self.target[test_idx]

        return X_train, y_train, X_test, y_test, training_idx, test_idx

    def separateSampleClass(self):

        # Discriminating the classes sample
        self.ind_class = []
        for i in xrange(0, len(self.classes)):
            self.ind_class.append(np.where(self.target == self.classes[i]))

    def getSampleClass(self):

        data = []
        target = []
        # Selecting the 'train sample' on the basis of the previously retrieved indices
        for i in xrange(0, len(self.classes)):
            data.append(self.data[self.ind_class[i]])
            target.append(self.target[self.ind_class[i]])

        return data, target

    def getIndClass(self):

        return self.ind_class