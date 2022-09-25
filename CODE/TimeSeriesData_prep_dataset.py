import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch


from skimage.util.shape import view_as_windows
import warnings



class TimeSeriesData_prep_dataset(Dataset):
    """
    Class for creating dataset using PyTorch data primitive Dataset. An instance of this class can be used in the 
    PyTorch data primitive Dataloader
    
    The following patameters can be adjusted:
    @windwo_size: Size of window which is used as Input in CNN
    @feature_of_interest: List of all features which should be used in the CNN
    @list_of_train_BSD_states: List of BSD states which should be used for training. Be careful at least 4 BSD
    states representing the 4 different classes should be included for the training
    @list_of_test_BSD_states: List of BSD states which should be used for testing
    """
    
    
    def __init__(self, data_path, window_size, overlap_size, numpy_array_names, features, features_of_interest):
        """
        Constructor

        INPUT:
        @data_path: path to dataset
        @window_size: defines number of samples per window
        @overlap_size: defines number of samples overlapping between neighbouring windows
        @numpy_array_names: defines the names of the .npy files which should be loaded as numpy array
        @features: all features which were generally recorded
        @features_of_interest: defines which features should be included
        """
        
        self.data_path = data_path
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.numpy_array_names = numpy_array_names
        self.features_of_interest = features_of_interest
        self.features = features
        self.x_data, self.y_data, self.n_samples = self.__get_sliced_data_and_labels()

    def split_data(self, x_data, y_data):
        """
        Split data in windows of equal size with overlap
        
        INPUT:
        @data: data numpy array of shape [elements per file, features]
        @window: number of elements per window
        @overlap_size: defines the overlapping elements between consecutive windows
        
        OUTPUT
        @data: data numpy array of shape [number_of_windows, elements per window, features]
        """

        y_data = np.expand_dims(y_data,axis = 1)
        if self.window_size==self.overlap_size:
            raise Exception("Overlap arg must be smaller than length of windows")
        S = self.window_size - self.overlap_size
        nd0 = ((len(x_data)-self.window_size)//S)+1
        if nd0*S-S!=len(x_data)-self.window_size:
            warnings.warn("Not all elements were covered")
        return view_as_windows(x_data, (self.window_size,x_data.shape[1]), step=S)[:,0,:,:], view_as_windows(y_data, (self.window_size, y_data.shape[1]), step=S)[:,0,:,:]

    def del_nan_element(self, data_with_nan, label_with_nan):
        """
        Delete all elements in the data which have any nan valued feature
        
        INPUT:
        @data_with_nan: data numpy array containing nan_values
        
        OUTPUT
        @data_with_nan: data numpy array inlcuding just elements per window which do have no nan_vaues in any feature
        """
        nan_val = np.isnan(data_with_nan) #mask for all nan_elements as 2d array [elements_per_window, features]
        nan_val = np.any(nan_val,axis = 1) #mask for all nan_rows as 1d array [elements_per_window]
        return data_with_nan[nan_val==False], label_with_nan[nan_val==False]

    def __get_sliced_data_and_labels(self):
        """
        preprocess and window data

        OUTPUT:
        @X_data: windowed and preprocessed data
        @y_data: windowed and preprocessed labels
        @n_samples: number of samples
        """
        X_data = np.load(os.path.join(self.data_path, self.numpy_array_names[0] + ".npy"))
        y_data = np.load(os.path.join(self.data_path, self.numpy_array_names[1] + ".npy"))


        feature_index_list = np.where(np.isin(self.features, self.features_of_interest)) #Get index for all features of interest
        X_data = X_data[:,feature_index_list] #Slice numpy array such that just features of interest are included
        X_data = np.squeeze(X_data, axis = 1) #Create one extra dimension for widows
        X_data, y_data = self.del_nan_element(X_data, y_data) #Delete all elements with a nan in any feature of interest
        X_data, y_data = self.split_data(X_data, y_data) #Window the data

        #just pick those windows which contain samples with equal labels
        array_to_check_y = np.tile(np.expand_dims(np.expand_dims(y_data[:,0,0],axis=1), axis=1),(np.shape(y_data)[1],1))
        mask = np.squeeze(np.all(array_to_check_y==y_data, axis = 1), axis=1)
        X_data = X_data[mask,:,:]
        y_data = y_data[mask,:,:]
        assert np.all(array_to_check_y[mask,:,:] == y_data)

        y_data = y_data[:,0,0] # one label for each window
        X_data = np.swapaxes(X_data,1,2) #swap axes for CNN

        n_samples = len(y_data)
        X_data = torch.from_numpy(X_data)
        y_data = torch.from_numpy(y_data)
    
        return X_data, y_data, n_samples

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        """
        Returns item of dataset by index

        OUTPUT:
        @self.x_data[index]: indexed sample
        @self.y_data[index]: indexed label
        """

        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        """
        Returns len of dataset

        OUTPUT:
        @self.n_samples: number of samples in dataset
        """

        return self.n_samples




