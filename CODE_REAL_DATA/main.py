import os
import sys
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import random
from datetime import datetime
import torch

from torch.utils.tensorboard import SummaryWriter

#variant1
from Dataloader import Dataloader

from Loss_CNN import Loss_CNN
from Classifier import Classifier
from MMD_loss import MMD_loss
from MMD_loss_CNN import MMD_loss_CNN

from CNN import CNN
from plotter import Plotter

from file_selector import File_selector

#variant2
from Preprocesser import Preprocessor
from TimeSeriesData_prep_dataset import TimeSeriesData_prep_dataset
from Dataloader_prep_dataset import Dataloader_prep_dataset


def main():
    
    #Unpack arguments for training
    train_params = sys.argv[1:]
    print(train_params)
    gpu_name = str(train_params[0])
    experiment_name = str(train_params[1]) #used in Tensorboard when using tracking several Experiments --> prevents overwriting data from previous experiment
    num_epochs = int(train_params[2]) #defines number of training epochs
    GAMMA = float(train_params[3]) #defines GAMMA value 
    GAMMA_reduction = float(train_params[4]) # can be used to decrease GAMMA each Epoch (GAMMA_next_epoch = GAMMA * GAMMA_reduction)
    num_pool = int(train_params[5]) #if 1 --> Pooling layer just after Conv1, if 2 --> Pooling layer just after Conv1 & Conv2 , if 3 --> Pooling layer just after Conv1 & Conv2 & Conv3
    MMD_layer_activation_flag = train_params[6:12] #defines which layer ins included in MMD Loss --> Bool List for layer order [Conv1, Conv2, Conv3, Flatten, FC1, FC2]
    MMD_layer_activation_flag = [eval(item.title()) for item in MMD_layer_activation_flag] #convert from String to Bool values

    #Read in variable lists with variable length

    len_feature_of_interest = 12 + int(train_params[12]) + 1
    features_of_interest = train_params[13:len_feature_of_interest] #Features of interest

    #len_list_of_source_BSD_states = len_feature_of_interest + int(train_params[len_feature_of_interest]) + 1
    #list_of_source_BSD_states = train_params[len_feature_of_interest+1:len_list_of_source_BSD_states] #Source BSD states

    #len_list_of_target_BSD_states = len_list_of_source_BSD_states + int(train_params[len_list_of_source_BSD_states]) + 1
    #list_of_target_BSD_states = train_params[len_list_of_source_BSD_states+1:len_list_of_target_BSD_states] #Target BSD states
    file_selector = File_selector()
    class_0_source, class_1_source, class_0_target, class_1_target = file_selector.select()

    list_of_source_BSD_states = [class_0_source, class_1_source]
    list_of_target_BSD_states = [class_0_target, class_1_target]

    print(list_of_source_BSD_states, list_of_target_BSD_states)

    len_class_0_labels = len_feature_of_interest + int(train_params[len_feature_of_interest]) + 1
    class_0_labels = train_params[len_feature_of_interest+1:len_class_0_labels] #Classes considered in unhealthy class

    len_class_1_labels = len_class_0_labels + int(train_params[len_class_0_labels]) + 1
    class_1_labels = train_params[len_class_0_labels+1:len_class_1_labels] #Classes considered in healthy class

    print(f"\nTRAINING PARAMETER: \n GPU Name: {gpu_name} \n Experiment Name: {experiment_name} \n Number of Epochs: {num_epochs} \n GAMMA: {GAMMA} \n GAMMA reduction: {GAMMA_reduction} \n number of Pooling Layers: {num_pool} \n Flags for layers inlcuded in MMD Loss: {MMD_layer_activation_flag} \n Features of Interest: {features_of_interest} \n Folders of considered Source BSD states: {list_of_source_BSD_states} \n Folders of considered Target BSD states: {list_of_target_BSD_states} \n Classes considered as unhealthy binary Class: {class_0_labels} \n Classes considered as healthy binary Class: {class_1_labels}")

    #Check if CUDA is available
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f"the device for executing the code is: {device}")

    #Create random seeds
    random_seed = random.randrange(0,100)

    #Folder name to store data for each experiment
    features_of_interest_folder = features_of_interest[0].replace("/", "_")
    features_of_interest_folder = features_of_interest_folder.replace(":", "_")
    date =  datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    folder_to_store_data = "experiments/feature=" + str(features_of_interest_folder)  + "_" + "GAMMA=" + str(GAMMA) + "_" +"GAMMA_reduction" + str(GAMMA_reduction) + "_" + "num_pool=" + str(num_pool) + "_" + str(MMD_layer_activation_flag) + "_" + date

    #Generate folder structure to store plots and data
    current_directory = os.getcwd()
    path_learning_curve = os.path.join(current_directory, folder_to_store_data, "learning_curve")
    path_learning_curve_data = os.path.join(current_directory, folder_to_store_data, "learning_curve_data")
    path_data_distribution = os.path.join(current_directory, folder_to_store_data, "data_distribution")
    path_data_distribution_data = os.path.join(current_directory, folder_to_store_data, "data_distribution_data")
    path_accuracy = os.path.join(current_directory, folder_to_store_data, "accuracy")
    path_best_model = os.path.join(current_directory, folder_to_store_data, "best_model")

    if not os.path.exists(path_learning_curve): #Folder to store Learning Curve Plots 
        os.makedirs(path_learning_curve)
    if not os.path.exists(path_learning_curve_data): #Folder to store Learning Curve Plots Data
        os.makedirs(path_learning_curve_data)
    if not os.path.exists(path_data_distribution): #Folder to store Data Distribuiton Plots 
        os.makedirs(path_data_distribution)
    if not os.path.exists(path_data_distribution_data): #Folder to store Data Distribuiton Plots Data 
        os.makedirs(path_data_distribution_data)
    if not os.path.exists(path_accuracy): #Folder to store Accuracies of Training
        os.makedirs(path_accuracy)
    if not os.path.exists(path_best_model): #Folder to store Accuracies of Training
        os.makedirs(path_best_model)


    #################
    #   Training    #
    #################

    #Init plotter for generating plots from data
    plotter = Plotter(folder_to_store_data)

    #Create csv file to store data 
    f_learning_curve = open(f'{folder_to_store_data}/learning_curve_data/learning_curve.csv', 'w')
    f_accuracy = open(f'{folder_to_store_data}/accuracy/accuracies.csv', 'w')

    #Create csv writer to store data
    f_learning_curve_writer = csv.writer(f_learning_curve)
    f_learning_curve_writer.writerow(['running_acc_source_val','running_acc_target_val','running_source_ce_loss_val','running_target_ce_loss_val','running_mmd_loss_val','running_acc_source_mmd','running_acc_target_mmd','running_source_ce_loss_mmd','running_target_ce_loss_mmd','running_mmd_loss_mmd','running_acc_source_ce','running_acc_target_ce','running_source_ce_loss_ce','running_target_ce_loss_ce','running_mmd_loss_ce'])

    #Header for csv file
    f_accuracy_writer = csv.writer(f_accuracy)
    f_accuracy_writer.writerow(['accuracy_source_val','accuracy_target_val','accuracy_source_mmd','accuracy_target_mmd','accuracy_source_ce','accuracy_target_ce'])

    #Init writer for tensorboard    
    writer_source_val = SummaryWriter(f'runs/{experiment_name}/source_val')
    writer_source_mmd = SummaryWriter(f'runs/{experiment_name}/source_mmd')
    writer_source_ce = SummaryWriter(f'runs/{experiment_name}/source_ce')
    writer_target_val = SummaryWriter(f'runs//{experiment_name}target_val')
    writer_target_mmd = SummaryWriter(f'runs/{experiment_name}/target_mmd')
    writer_target_ce = SummaryWriter(f'runs/{experiment_name}/target_ce')

    writer_source = {}
    writer_source["val"] = writer_source_val
    writer_source["mmd"] = writer_source_mmd
    writer_source["ce"] = writer_source_ce

    writer_target = {}
    writer_target["val"] = writer_target_val
    writer_target["mmd"] = writer_target_mmd
    writer_target["ce"] = writer_target_ce

    #Training iterations
    phases = ['val', 'mmd', 'ce']

    #Windowing details
    window_size = 1024
    overlap_size = 0

    #Path where dataset is stored
    data_path = Path(os.getcwd()).parents[1]
    data_path = os.path.join(data_path, "data")

    #Dataloader split of data
    dataloader_split_ce = 0.4
    dataloader_split_mmd = 0.2
    dataloader_split_val = 0.2

    #Batch size
    batch_size = 32
    
    
    ###Dataloader Variant 1####
    
    dataloader_source = Dataloader(data_path, list_of_source_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed, class_0_labels, class_1_labels)
    dataloader_target = Dataloader(data_path, list_of_target_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed, class_0_labels, class_1_labels)
    source_loader = dataloader_source.create_dataloader()
    target_loader = dataloader_target.create_dataloader()
    
    
    ### Dataloader Variant 2###
    """
    
    #Names of numpy arrays where data is stored --> stored as .npy
    source_numpy_array_names = ["source_X", "source_y"]
    target_numpy_array_names = ["target_X", "target_y"]

    #Preprocess data
    preprocessor_source = Preprocessor(data_path, list_of_source_BSD_states, window_size, overlap_size, features_of_interest, source_numpy_array_names, class_0_labels, class_1_labels)
    preprocessor_target = Preprocessor(data_path, list_of_target_BSD_states, window_size, overlap_size, features_of_interest, target_numpy_array_names, class_0_labels, class_1_labels)

    #Concatenate the preprocessed Data from all csv files
    features  = preprocessor_source.concatenate_data_from_BSD_state()
    _ = preprocessor_target.concatenate_data_from_BSD_state()

    #Create Dataset
    dataset_source = TimeSeriesData_prep_dataset(data_path, window_size, overlap_size, source_numpy_array_names, features, features_of_interest)
    dataset_target = TimeSeriesData_prep_dataset(data_path, window_size, overlap_size, target_numpy_array_names, features, features_of_interest)

    #create Dataloader
    dataloader_source = Dataloader_prep_dataset(dataset_source, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed)
    dataloader_target = Dataloader_prep_dataset(dataset_target, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed)
    source_loader = dataloader_source.create_dataloader()
    target_loader = dataloader_target.create_dataloader()
    """
    

    #define Sigma for RBF Kernel in MMD Loss
    SIGMA = torch.tensor([1,2,4,8,16],dtype=torch.float64)

    #Define mmd_loss_flag to specify which trainingphases use MMD and CE Loss
    MMD_loss_flag_phase = {}
    MMD_loss_flag_phase["val"] = False
    MMD_loss_flag_phase["mmd"] = True
    MMD_loss_flag_phase["ce"] = False

    #Model
    input_size = len(features_of_interest)
    hidden_fc_size_1 = 50
    hidden_fc_size_2 = 3
    output_size = 2
    model_cnn = CNN(input_size, hidden_fc_size_1, num_pool, window_size, random_seed)
    model_fc = Classifier(hidden_fc_size_1, hidden_fc_size_2, output_size, random_seed)

    #models to gpu if available
    model_cnn = model_cnn.to(device)
    model_fc = model_fc.to(device)
    
    if (next(model_cnn.parameters()).is_cuda and next(model_fc.parameters()).is_cuda):
        print("Models are on GPU!!")

    #Loss
    criterion = torch.nn.CrossEntropyLoss().to(device)
    MMD_loss_calculator = MMD_loss(fix_sigma = SIGMA).to(device)
    MMD_loss_CNN_calculator = MMD_loss_CNN(fix_sigma = SIGMA).to(device)
    loss_cnn = Loss_CNN(model_cnn, model_fc, criterion, MMD_loss_calculator, MMD_loss_CNN_calculator, MMD_layer_activation_flag)

    #Optimizer
    optimizer1 = torch.optim.Adam([
    {'params': model_cnn.parameters()},
    {'params': model_fc.parameters(), 'lr': 1e-4}
    ], lr=1e-2, betas=(0.9, 0.999))

    optimizer2 = torch.optim.Adam(model_fc.parameters(), lr=1e-2, betas=(0.9, 0.999))

    #Safe the random seed as txt file
    f_random_seed = open(f'{folder_to_store_data}/best_model/random_seed.txt', 'w')
    f_random_seed.write(str(random_seed))
    f_random_seed.close()

    #Safe the Model and Training hyperparameter as txt file
    f_hyperparameter = open(f'{folder_to_store_data}/best_model/hyperparameter.txt', 'w')
    f_hyperparameter.write(f'features of interest: {features_of_interest}\n')
    f_hyperparameter.write(f'num_epochs: {num_epochs}\n')
    f_hyperparameter.write(f'GAMMA: {GAMMA}\n')
    f_hyperparameter.write(f'GAMMA_reduction: {GAMMA_reduction}\n')
    f_hyperparameter.write(f'num_pool: {num_pool}\n')
    f_hyperparameter.write(f'MMD_layer_flag: {MMD_layer_activation_flag}\n')
    f_hyperparameter.write(f'list_of_source_BSD_states: {list_of_source_BSD_states}\n')
    f_hyperparameter.write(f'list_of_target_BSD_states: {list_of_target_BSD_states}\n')
    f_hyperparameter.write(f'dataloader_split_ce: {dataloader_split_ce}\n')
    f_hyperparameter.write(f'dataloader_split_mmd: {dataloader_split_mmd}\n')
    f_hyperparameter.write(f'dataloader_split_val: {dataloader_split_val}\n')
    f_hyperparameter.write(f'batch_size: {batch_size}\n')
    f_hyperparameter.write(f'input_size_CNN: {input_size}\n')
    f_hyperparameter.write(f'hidden_fc_size_1: {hidden_fc_size_1}\n')
    f_hyperparameter.write(f'hidden_fc_size_2: {hidden_fc_size_2}\n')
    f_hyperparameter.write(f'output_size_FC: {output_size}\n')
    f_hyperparameter.write(f'SIGMA: {SIGMA}\n')
    f_hyperparameter.write(f'criterion: {criterion}\n')
    f_hyperparameter.write(f'optimizer1: {optimizer1}\n')
    f_hyperparameter.write(f'optimizer2: {optimizer2}\n\n')
    f_hyperparameter.write(f'Model CNN: {model_cnn}\n\n')
    f_hyperparameter.write(f'Model FC: {model_fc}') 
    f_hyperparameter.close()



    #Init variables which collect loss, accuracies for each epoch and train phase
    source_ce_loss_collected = 0
    target_ce_loss_collected = 0
    mmd_loss_collected = 0
    acc_total_source_collected = 0
    acc_total_target_collected = 0
    balanced_target_accuracy_collected = 0

    #Store data about best performing model (balanced accuracy on validation set)
    max_target_val_accuracy = 0
    best_GAMMA = None
    best_features_of_interest = None
    best_pool = None


    # Train and Validate the model
    for epoch in range(num_epochs):

        #Reduce the GAMMA in each epoch
        GAMMA*=GAMMA_reduction

        #Init array which collects the data in FC2 for plottnig the 3-dimensional data distribution
        class_0_source_fc2_collect = torch.empty((0,3))
        class_1_source_fc2_collect = torch.empty((0,3))
        class_0_target_fc2_collect = torch.empty((0,3))
        class_1_target_fc2_collect = torch.empty((0,3))

        #Init/reset list to collect data which is stored in one csv file line
        f_accuracy_collect = []
        learning_curve_data_collect = []

        #iterate through phases
        for phase in phases:

            #init the dataloader for source and target data for each epoch
            iter_loader_source = iter(source_loader[phase])
            iter_loader_target = iter(target_loader[phase])

            #iterate through batches of phase specific dataloader
            for _ in range(len(iter_loader_source)):
                
                ########Forward pass########
                batch_data_source, labels_source = iter_loader_source.next() #Windows and labels from source domain (dim = Batch Size)
                batch_data_target, labels_target = iter_loader_target.next() #Windows and labels from target domain (dim = Batch Size)
                batch_data = torch.cat((batch_data_source, batch_data_target), dim=0) #Conncatination of windows from Source and Target Domain (dim = Batch Size*2)

                labels_source = labels_source.to(device) #Data to GPU if availbl
                labels_target = labels_target.to(device) #Data to GPU if availbl
                batch_data = batch_data.to(device) #Data to GPU if availble
                
                #if batch_data.is_cuda and labels_source.is_cuda and labels_target.is_cuda:
                #    print("Samples are all on GPU !!")

                #Validation
                if phase == "val":
                    model_cnn.train(False)
                    model_fc.train(False)
                    
                    with torch.no_grad():
                        _, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, balanced_target_accuracy, class_0_source_fc2, class_1_source_fc2, class_0_target_fc2, class_1_target_fc2 = loss_cnn.forward(batch_data, labels_source, labels_target, MMD_loss_flag_phase[phase], GAMMA)
                        
                        #Collect latent features of fc2 for plot 
                        class_0_source_fc2_collect = torch.cat((class_0_source_fc2_collect, class_0_source_fc2.to("cpu")), 0)
                        class_1_source_fc2_collect = torch.cat((class_1_source_fc2_collect, class_1_source_fc2.to("cpu")), 0)
                        class_0_target_fc2_collect = torch.cat((class_0_target_fc2_collect, class_0_target_fc2.to("cpu")), 0)
                        class_1_target_fc2_collect = torch.cat((class_1_target_fc2_collect, class_1_target_fc2.to("cpu")), 0)

                #Training
                else:
                    model_cnn.train(True)
                    model_fc.train(True)
                    
                    ######## Forward pass ########
                    loss, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, _, _, _, _, _ = loss_cnn.forward(batch_data, labels_source, labels_target, MMD_loss_flag_phase[phase], GAMMA)
                    
                    #Detatch losses which are just used for plotting (detatch is necessary to store data) not for Training --> Just loss is used in backward pass
                    mmd_loss = mmd_loss.detach()
                    source_ce_loss = source_ce_loss.detach()
                    target_ce_loss = target_ce_loss.detach()

                    ######## Backward pass ########
                    if phase == "mmd": # Phase 1 Optimization whole NN with CE + GAMMA * MMD
                        optimizer1.zero_grad()
                        loss.backward()
                        optimizer1.step()
                    elif phase == "ce": # Phase 2 Optimization FC2 & FC3 with CE 
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer2.step()

                #collect loss, accuracies for each epoch
                mmd_loss_collected += mmd_loss.item()
                source_ce_loss_collected += source_ce_loss.item()
                target_ce_loss_collected += target_ce_loss.item()
                acc_total_source_collected += acc_total_source
                acc_total_target_collected += acc_total_target
                balanced_target_accuracy_collected += balanced_target_accuracy
            
            # store data distribution in FC2 for specific epochs and during Validation Phase in csv
            if phase == "val" and (epoch ==0 or epoch ==20 or epoch == 40 or epoch ==80):
                
                df1 = pd.DataFrame({'class_0_source_fc2_collect_0_dim':class_0_source_fc2_collect[:, 0]})
                df2 = pd.DataFrame({'class_0_source_fc2_collect_1_dim':class_0_source_fc2_collect[:, 1]})
                df3 = pd.DataFrame({'class_0_source_fc2_collect_2_dim':class_0_source_fc2_collect[:, 2]})
                df4 = pd.DataFrame({'class_1_source_fc2_collect_0_dim':class_1_source_fc2_collect[:, 0]})
                df5 = pd.DataFrame({'class_1_source_fc2_collect_1_dim':class_1_source_fc2_collect[:, 1]})
                df6 = pd.DataFrame({'class_1_source_fc2_collect_2_dim':class_1_source_fc2_collect[:, 2]})
                df7 = pd.DataFrame({'class_0_target_fc2_collect_0_dim':class_0_target_fc2_collect[:, 0]})
                df8 = pd.DataFrame({'class_0_target_fc2_collect_1_dim':class_0_target_fc2_collect[:, 1]})
                df9 = pd.DataFrame({'class_0_target_fc2_collect_2_dim':class_0_target_fc2_collect[:, 2]})
                df10 = pd.DataFrame({'class_1_target_fc2_collect_0_dim':class_1_target_fc2_collect[:, 0]})
                df11 = pd.DataFrame({'class_1_target_fc2_collect_1_dim':class_1_target_fc2_collect[:, 1]})
                df12 = pd.DataFrame({'class_1_target_fc2_collect_2_dim':class_1_target_fc2_collect[:, 2]})
                pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12],axis=1).to_csv(f'{folder_to_store_data}/data_distribution_data/data_distribution_{epoch}.csv', index = False)

            # Normalize collected loss, accuracies with number of batches for each completed train phase
            running_mmd_loss = mmd_loss_collected / len(source_loader[phase])
            running_acc_source = acc_total_source_collected / len(source_loader[phase])
            running_acc_target = acc_total_target_collected / len(target_loader[phase])
            running_source_ce_loss = source_ce_loss_collected / len(source_loader[phase])
            running_target_ce_loss = target_ce_loss_collected / len(target_loader[phase])
            running_balanced_target_accuracy = balanced_target_accuracy_collected / len(target_loader[phase])

            #In each epoch check the average target accuracy of the model on the Traget Domain validation dataset and store model and Training Parameters if it performed better than before
            if phase == "val":
                if max_target_val_accuracy < running_balanced_target_accuracy: #if balanced Target Domain Validation Dataset Accuracy is better than ever seen before
                    max_target_val_accuracy = running_balanced_target_accuracy #safe best seen Target Domain Validation Dataset Accuracy
                    torch.save(model_cnn.state_dict(), f'{folder_to_store_data}/best_model/model_cnn.pt') #Save CNN 
                    torch.save(model_fc.state_dict(), f'{folder_to_store_data}/best_model/model_fc.pt') #Save Classifier
                    #Safe the Parameters of the best performing Model 
                    best_GAMMA = GAMMA
                    best_features_of_interest = features_of_interest
                    best_pool = num_pool

            #Add train data to tensorboard list
            writer_source[phase].add_scalar(f'accuracy', running_acc_source, epoch)
            writer_target[phase].add_scalar(f'accuracy', running_acc_target, epoch)
            writer_source[phase].add_scalar(f'ce_loss', running_source_ce_loss, epoch)
            writer_target[phase].add_scalar(f'ce_loss', running_target_ce_loss, epoch)
            writer_source[phase].add_scalar(f'mmd_loss', running_mmd_loss, epoch)

            #collect data which is stored in one line of csv --> each row contains Source and Target Domain Accuracy, CE-Loss, MMD-Loss for each Phase
            learning_curve_data_collect = learning_curve_data_collect + [running_acc_source, running_acc_target, running_source_ce_loss, running_target_ce_loss, running_mmd_loss]
            f_accuracy_collect = f_accuracy_collect + [running_acc_source, running_acc_target]

            #Reset variable for collected loss, accuracies for each completed train phase
            source_ce_loss_collected = 0
            target_ce_loss_collected = 0
            mmd_loss_collected = 0
            acc_total_source_collected = 0
            acc_total_target_collected = 0
            balanced_target_accuracy_collected = 0

        # After completing all phases for the epoch --> Write Source and Target Domain Accuracy, CE-Loss, MMD-Loss for each Phase in row of CSV file
        f_learning_curve_writer.writerow(learning_curve_data_collect)
        f_accuracy_writer.writerow(f_accuracy_collect)
        
        print(f"Epoch {epoch+1}/{num_epochs} successfull")

    #close csv writer for accuracy and learning curves
    f_accuracy.close()
    f_learning_curve.close()

    #plot learning curves and data distribtuion from csv files stored during Training as CSV
    plotter.plot_distribution()
    plotter.plot_curves()

    print(f"With an Accuracy of: {max_target_val_accuracy}, the model with the following hyperparameter performed best:\nbest_features_of_interest: {best_features_of_interest}\nbest_GAMMA: {best_GAMMA}\nbest_pool: {best_pool}")







    ################
    #   Testing    #
    ################

    #Load CNN for Testing from stored parameters of best performing Model during Training measured with Traget Domain validation dataset
    model_cnn_test =  CNN(input_size, hidden_fc_size_1, num_pool, window_size, random_seed)
    model_cnn_test.load_state_dict(torch.load(f'{folder_to_store_data}/best_model/model_cnn.pt'))
    model_cnn_test.eval()

    #Load Classifier for Testing from stored parameters of best performing Model during Training measured with Traget Domain validation dataset
    model_fc_test = Classifier(hidden_fc_size_1, hidden_fc_size_2, output_size, random_seed)
    model_fc_test.load_state_dict(torch.load(f'{folder_to_store_data}/best_model/model_fc.pt'))
    model_fc_test.eval() 

    #Init collector for accuracy for each batch
    acc_total_target_test_collected = 0

    test_loader_target = iter(target_loader["test"])

    #iterate through batches of the dataloader for the target domain test dataset
    for _ in range(len(test_loader_target)):
                
        ########Forward pass########
        batch_data_target_test, labels_target_test = test_loader_target.next() # windows and labels for the target domain test dataset
        _, _, _, _, x_fc1_test = model_cnn_test(batch_data_target_test.float())
        _, x_fc3_test = model_fc_test(x_fc1_test)

        #Average target domain test dataset accuracy for all samples in Batch
        argmax_target_test_pred = torch.argmax(x_fc3_test[:batch_size, :], dim=1)
        result_target_test_pred = argmax_target_test_pred == labels_target_test
        correct_target_test_pred = result_target_test_pred[result_target_test_pred == True]
        acc_total_target_test = 100 * len(correct_target_test_pred)/len(labels_target_test)
        acc_total_target_test_collected += acc_total_target_test

    #Average target domain test dataset accuracy for all samples in dataset
    running_acc_target_test = acc_total_target_test_collected / len(test_loader_target)

    #Safe target domain test dataset accuracy
    f_target_test_accuracy = open(f'{folder_to_store_data}/best_model/target_test_accuracy.txt', 'w')
    f_target_test_accuracy.write(str(running_acc_target_test))
    f_target_test_accuracy.close()

if __name__ == "__main__":
    main()
