import os
import sys
import pickle

import numpy as np

class DataSet(object):
    """Class Object encompassing the time series dataset
    
    """
    def __init__(self, fpath, need_shuffle=True):
        with open(fpath, 'rb') as f:
            self.wholeset = pickle.load(f, encoding='latin1')
        
        self.data = self.wholeset['data']
        self.target = self.wholeset['val']
        self.aux = self.wholeset['aux']

        self._num_data = len(self.data)

        # dimensions regarding input data
        self._len_input = self.data.shape[2]
        self._dim_input = self.data.shape[1]

        self._shuffle = need_shuffle
        self._indices = np.arange(self._num_data)

        # variables regarding iteration
        self._index_epoch = 0


    def iter_once(self, batch_size):
        self._index_epoch = 0
        
        if self._shuffle:
            np.random.shuffle(self._indices)

        while True :
            start = self._index_epoch
            self._index_epoch += batch_size

            if self._index_epoch > self._num_data :
                end = self._num_data
                indices = self._indices[start:end]
                if start < self._num_data:
                    yield self.data[indices], self.target[indices], self.aux[indices]
                break

            end = self._index_epoch
            indices = self._indices[start:end]
            yield self.data[indices], self.target[indices], self.aux[indices]
