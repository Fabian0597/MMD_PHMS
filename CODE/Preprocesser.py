import os
import csv
from venv import create
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch


from skimage.util.shape import view_as_windows
import warnings


class Preprocessor:
    """
    The following patameters can be adjusted:
    @windwo_size: Size of window which is used as Input in CNN
    @feature_of_interest: List of all features which should be used in the CNN
    @list_of_train_BSD_states: List of BSD states which should be used for training. Be careful at least 4 BSD
    states representing the 4 different classes should be included for the training
    @list_of_test_BSD_states: List of BSD states which should be used for testing
    """
    
    
    def __init__(self, data_path, list_of_BSD_states, window_size, overlap_size, features_of_interest, numpy_array_names, class_0_labels, class_1_labels):
        
        self.data_path = data_path
        self.list_of_BSD_states =list_of_BSD_states
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.features_of_interest = features_of_interest
        self.numpy_array_names = numpy_array_names
        self.class_0_labels = class_0_labels
        self.class_1_labels = class_1_labels

    def create_folder_dictionary(self):
        """
        Create a dictionaty for testing and training containing folder names as keys and files as values
        
        INPUT:
        @list_of_train_BSD_states: list containing the training BSD states as string
        @list_of_test_BSD_states: list containing the testing BSD states as string
        @data_path: data directory containing folders for each BSD state
        
        OUTPUT
        @training_folders: dictionary folders and keys for training
        @testing_folders: dictionary folders and keys for testing
        """
        
        state_dictionary = {
            "1":"NR01_20200317_PGS_31_BSD_31",
            "2":"NR02_20200423_PGS_31_BSD_21",
            "3":"NR03_20200424_PGS_31_BSD_11",
            "4":"NR04_20200424_PGS_31_BSD_P1",
            "5":"NR05_20200930_PGS_31_BSD_22",
            "6":"NR06_20201001_PGS_31_BSD_12",
            "7":"NR07_20201001_PGS_31_BSD_32",
            "8":"NR08_20200918_PGS_31_BSD_33",
            "9":"NR09_20200917_PGS_31_BSD_P2",
            "10":"NR10_20200502_PGS_21_BSD_31",
            "11":"NR11_20200429_PGS_21_BSD_21",
            "12":"NR12_20200429_PGS_21_BSD_11",
            "13":"NR13_20200428_PGS_21_BSD_P1",
            "14":"NR14_20200731_PGS_21_BSD_22",
            "15":"NR15_20200901_PGS_21_BSD_12",
            "16":"NR16_20200908_PGS_21_BSD_32",
            "17":"NR17_20200717_PGS_21_BSD_33",
            "18":"NR18_20200714_PGS_21_BSD_P2",
            "19":"NR19_20200505_PGS_11_BSD_31",
            "20":"NR20_20200507_PGS_11_BSD_21",
            "21":"NR21_20200508_PGS_11_BSD_11",
            "22":"NR22_20200508_PGS_11_BSD_P1",
            "23":"NR23_20200511_PGS_11_BSD_22",
            "24":"NR24_20200512_PGS_11_BSD_12",
            "25":"NR25_20200512_PGS_11_BSD_32",
            "26":"NR26_20200513_PGS_11_BSD_33",
            "27":"NR27_20200513_PGS_11_BSD_P2",
        }
        
        
        data_folders = {}
        
        for element in self.list_of_BSD_states: #iterate through BSD_states which should be considerd
            data_folders[state_dictionary[element]]=os.listdir(os.path.join(self.data_path,state_dictionary[element])) #create dictionary with BSD_state folder names as keys and files contained in that foulder as values
        return data_folders

    def get_features(self, data_folders):
        """
        Creates a list of all feature names

        INPUT:
        @path: path to any BSD file since the features are the same for all files
        
        OUTPUT
        @features: list of features:
        ['C:s_ist/X', 'C:s_soll/X', 'C:s_diff/X', 'C:v_(n_ist)/X', 'C:v_(n_soll)/X', 'C:P_mech./X', 'C:Pos._Diff./X',
        'C:I_ist/X', 'C:I_soll/X', 'C:x_bottom', 'C:y_bottom', 'C:z_bottom', 'C:x_nut', 'C:y_nut', 'C:z_nut',
        'C:x_top', 'C:y_top', 'C:z_top', 'D:s_ist/X', 'D:s_soll/X', 'D:s_diff/X', 'D:v_(n_ist)/X', 'D:v_(n_soll)/X',
        'D:P_mech./X', 'D:Pos._Diff./X', 'D:I_ist/X', 'D:I_soll/X', 'D:x_bottom', 'D:y_bottom', 'D:z_bottom',
        'D:x_nut', 'D:y_nut', 'D:z_nut', 'D:x_top', 'D:y_top', 'D:z_top', 'S:x_bottom', 'S:y_bottom', 'S:z_bottom',
        'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]',
        'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]',
        'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']
        """

        #get the path of the first file from the first folder of considered BSD_states  
        first_BSD_path = list(data_folders.keys())[0] 
        first_file_path = data_folders[first_BSD_path][0]
        path_BSD_file = os.path.join(self.data_path, first_BSD_path, first_file_path) 

        #read the first line of csv file which conatin all feature names
        with open(path_BSD_file, 'r') as file:
            csvreader = csv.reader(file)
            features = next(csvreader)
        return features

    def concatenate_data_from_BSD_state(self):
        """
        Concatenates all the windowed data from each file to one big torch array
        
        INPUT:
        @folders: dictionary containing folders (as keys) and files (as values) to downloaded
        @data_path: data directory containing folders for each BSD state
        @features_of_interest: list of features which should be included for training
        @window_size: number of elements per widow
        
        OUTPUT:
        @n_samples: number of total elements from all included files
        @x_data: torch array containing all the data elements 
        @y_data: torch array containing the labels for all elements
        """
    
        # arrays to collect data and label
        x_data_concatenated = None
        y_data_concatenated = None

        data_folders = self.create_folder_dictionary()
        features = self.get_features(data_folders)
        
        iterator = 0
        first = True #in first iteration of conatenation make a deepcopy and afterwards concatenate arrays

        #If data was already loaded previously then just use the generated X_data.npy and y_data.npy files
        if os.path.isfile(os.path.join(self.data_path, self.numpy_array_names[0])+".npy") and os.path.isfile(os.path.join(self.data_path, self.numpy_array_names[1])+".npy"):
            print (f"CSV files: are already saved as numpy array {self.numpy_array_names[0]} and {self.numpy_array_names[1]}")
            return features

        #If data was not loaded previously then generate X_data.npy and y_data.npy files
        for BSD_path in data_folders.keys(): #folder path
            for file_path in data_folders[BSD_path]: #file path 
                path_BSD_file = os.path.join(self.data_path, BSD_path, file_path) # concatenate the data_path, folder and file path

                #get numpy array from CSV
                data_BSD_file = np.genfromtxt(path_BSD_file, dtype = np.dtype('d'), delimiter=',')[1:,:] #write csv in numpy array
                
                #rewrite labels as BSD_condition_1 = 1, BSD_condition_2 = 2, BSD_condition_3 = 3, BSD_condition_P1 = P
                label_file = BSD_path[-2] #take the first number of the BSD state for class label

                #Define binary classification
                if label_file in self.class_0_labels: #unhealthy class
                    label = 0
                elif label_file in self.class_1_labels: #healthy class
                    label = 1

                
                #concatenate the data from each file in one numpy array
                if  first == True: #overwrite variable
                    x_data_concatenated = np.copy(data_BSD_file)
                    y_data_concatenated = np.copy(np.asarray([label]*np.shape(data_BSD_file)[0]))
                    first = False
                else: #concatenate data numpy arrays
                    x_data_concatenated = np.concatenate((x_data_concatenated, data_BSD_file), axis=0)
                    y_data_concatenated = np.concatenate((y_data_concatenated,np.asarray([label]*np.shape(data_BSD_file)[0])), axis=0)
                
                iterator +=1
                print(f"{iterator}/{len(data_folders.keys())*len(data_folders[list(data_folders.keys())[0]])} folders downloaded")
                print(f"downloaded folder: {BSD_path}/{file_path}")
                print(f"Shape of collected datafram: X_shape: {np.shape(x_data_concatenated)}, Y_shape: {np.shape(y_data_concatenated)}")
        

        #Save data and label array
        n_samples = np.shape(x_data_concatenated)[0]
        x_data = x_data_concatenated
        y_data = y_data_concatenated
        np.save(os.path.join(self.data_path, self.numpy_array_names[0]), x_data)
        np.save(os.path.join(self.data_path, self.numpy_array_names[1]), y_data)

        return features
