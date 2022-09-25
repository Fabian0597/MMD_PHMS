from torch.utils.data import DataLoader
import torch
from TimeSeriesData import TimeSeriesData



class Dataloader():
    def __init__(self,data_path, list_of_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed, class_0_labels, class_1_labels) -> None:
        """
        Constructor

        INPUT:
        @data_path: path to dataset
        @list_of_BSD_states: list of BSD states which should be included in dataset
        @window_size: defines number of samples per window
        @overlap_size: defines number of samples overlapping between neighbouring windows
        @features_of_interest: defines which features should be included
        @dataloader_split_ce: split of dataset used for Training phase 2 (just CE-Loss)
        @dataloader_split_mmd: split of dataset used for Training phase 1 (MMD-Loss & CE-Loss)
        @dataloader_split_val: split of dataset used for Validation phase
        @batch_size: number of samples per batch
        @random_seed: random seed for dataset split
        @class_0_labels: classes considered in unhealthy class
        @class_1_labels: classes considered in healthy class
        """
        
        self.data_path = data_path
        self.list_of_BSD_states = list_of_BSD_states
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.features_of_interest = features_of_interest
        self.dataloader_split_ce = dataloader_split_ce
        self.dataloader_split_mmd = dataloader_split_mmd
        self.dataloader_split_val = dataloader_split_val
        self.batch_size = batch_size
        self.class_0_labels = class_0_labels
        self.class_1_labels = class_1_labels

        torch.manual_seed(0)

    def create_dataloader(self):
        """
        Create dataloader for training, validation, testing
    
        INPUT:
        @dataset_train: dataset with samples for training and validation from training domain
        @dataset_test: dataset with samples for testing from testing domain
        OUTPUT
        @data_loader: dictionary which contains the dataloader for training and val. Dataloaders can be accesses with keys "train", "val"
        @test_loader: dataloader for testing
        """

        #get dataset
        dataset = TimeSeriesData(self.data_path, self.list_of_BSD_states, self.window_size, self.overlap_size, self.features_of_interest, self.class_0_labels, self.class_1_labels)

        #define split between trainign phases, validation and testing
        dataset_size_ce= int(self.dataloader_split_ce * len(dataset))
        dataset_size_mmd= int(self.dataloader_split_mmd * len(dataset))
        dataset_size_val= int(self.dataloader_split_val * len(dataset))
        dataset_size_test = len(dataset) - dataset_size_ce - dataset_size_mmd - dataset_size_val

        #split dataset randomly
        dataset_train_ce, dataset_train_mmd, dataset_val, dataset_test = torch.utils.data.random_split(dataset, [dataset_size_ce, dataset_size_mmd, dataset_size_val, dataset_size_test])

        #dataloader
        train_ce_loader = DataLoader(dataset=dataset_train_ce,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)

        train_mmd_loader = DataLoader(dataset=dataset_train_mmd,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)


        val_loader = DataLoader(dataset=dataset_val,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)

        test_loader = DataLoader(dataset=dataset_test,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)

        data_loader = {}
        data_loader['ce'] = train_ce_loader
        data_loader['mmd'] = train_mmd_loader
        data_loader['val'] = val_loader
        data_loader['test'] = test_loader

        return data_loader