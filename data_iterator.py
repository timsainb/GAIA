import numpy as np
import queue, threading
import h5py
import warnings

class IteratorError(Exception):
    pass

class HDF5_iterator():
    def __init__(self, files, data_names, batch_size=16,queue_size=2,p_test=0.05, p_valid=0.05, multithreading=False):
        """
        Warning: there appears to be a memory leak in the multithreading implementation
        """
        self.multithreading = multithreading
        self.data_names = data_names # the names of the datasets to grab from the file
        self.files = files # the name of the file
        self.batch_size = batch_size # batch size for data we're grabbing
        if multithreading:
            warnings.warn("Multithreading is not fully implemented yet.")
            self.data_queue = queue.Queue(queue_size) # a queue with two space for two "chunks"
            self.sentinel = object() # this object is returned when the iterator is empty
        # the files we're opening
        self.hfs = {}
        for file in self.files:
            self.hfs[file] = h5py.File(file, 'r')
        print(self.data_names)
        # generate training/validation/testing_indexes
        self.train_idxs = np.random.permutation(np.concatenate([(np.repeat(file, len(self.hfs[file][self.data_names[0]])),
                                          np.arange(0, len(self.hfs[file][self.data_names[0]]))) for file in self.files],
                                              axis = 1).T)

        if p_test+p_valid > 0:
            self.test_idxs = self.train_idxs[:int(p_test*len(self.train_idxs))]
            self.valid_idxs = self.train_idxs[-int(p_valid*len(self.train_idxs)):]
            self.train_idxs = self.train_idxs[int(p_test*len(self.train_idxs)):-int(p_valid*len(self.train_idxs))]

    def new_epoch(self):
        """ Start a new epoch of iterations"""
        if self.multithreading:
            threading.Thread(target=self.load_task).start() # start iterations running in a second thread
        else:
            self.train_idxs = np.random.permutation(self.train_idxs)
            self.batch_idx = 0
            self.max_batch_size = len(self.train_idxs)/self.batch_size

    def load_hdf5(self, hf, cur_idxs):
        if len(self.data_names) > 1: # if there is more than one thing you're looking for
            #return [data[list(np.sort(cur_idxs))][np.argsort(np.argsort(np.array(cur_idxs).astype('int')))] for data in [hf[dn] for dn in self.data_names]]
            return [data[list(np.sort(cur_idxs))][np.argsort(np.argsort(np.array(cur_idxs).astype('int')))] for data in [hf[dn] for dn in self.data_names]]
        else: # if there's only one thing you're returning
            return [hf[self.data_names[0]][list(np.sort(cur_idxs))][np.argsort(np.argsort(np.array(cur_idxs).astype('int')))]]

    def load_idxs(self, cur_idxs):
        # get data for each file, then group it together
        loaded_data = [self.load_hdf5(self.hfs[file], cur_idxs[cur_idxs[:,0] == file,1]) for file in self.files if np.sum(cur_idxs[:,0] == file) > 0]
        return [np.concatenate([file_out[data_name] for file_out in loaded_data]) for data_name in range(len(self.data_names))]

    def load_task(self):
        # randomized permutations of data
        idxs = np.random.permutation(np.concatenate([(np.repeat(file, len(self.hfs[file][self.data_names[0]])),
                                          np.arange(0, len(self.hfs[file][self.data_names[0]]))) for file in self.files], axis = 1).T)
        for batch_idx in np.arange(0, len(idxs)-self.batch_size, self.batch_size): # iterate through batches
            cur_idxs = idxs[batch_idx:batch_idx+self.batch_size] # get current batch index
            self.data_queue.put(self.load_idxs(cur_idxs)) # add batch to cue
        self.data_queue.put(self.sentinel, True) # when there is no more data, add sentinel as an end marker

    def iterate(self):
        if self.multithreading:
            batch = self.data_queue.get(True)
            self.data_queue.task_done()
            if batch is self.sentinel:
                raise IteratorError("No more batches")
            return batch
        else:
            cur_idxs = self.train_idxs[(self.batch_idx*self.batch_size):(self.batch_idx*self.batch_size)+self.batch_size] # get current batch index
            if self.batch_idx >=self.max_batch_size:
                raise IteratorError("No more batches")
            self.batch_idx +=1
            return self.load_idxs(cur_idxs)
