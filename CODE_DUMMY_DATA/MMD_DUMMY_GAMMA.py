import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random 
import pandas as pd
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.utils.tensorboard import SummaryWriter

import csv


def create_curve(freq, ampl, freq_noise, ampl_noise, window_size):
    freq = random.gauss(freq, freq_noise) #add noise to frequency
    ampl = random.gauss(ampl, ampl_noise) #add noise to amplitude
    time = np.linspace(0, 10*np.pi, window_size)
    x = ampl*np.cos(freq*time)
    noise = np.random.normal(random.uniform(-1,0), random.uniform(0,1), window_size)
    x+=noise
    x = np.expand_dims(x, axis = 0) #expand to get 2d array (features, window length)
    x = np.expand_dims(x, axis = 0) #expand to get 3d array to store 2d elements
    #print(f"freq: {freq}, ampl:{ampl}")
    #plt.plot(time, x[0,0,:])
    #plt.show()
    
    return x

def create_data_window(n, frequencies, amplitudes, freq_noise, ampl_noise, window_size):
    
    X_data_class_1_domain_1 = create_curve(frequencies[0], amplitudes[0], freq_noise, ampl_noise, window_size)
    X_data_class_2_domain_1 = create_curve(frequencies[1], amplitudes[1], freq_noise, ampl_noise, window_size)
    X_data_class_1_domain_2 = create_curve(frequencies[2], amplitudes[2], freq_noise, ampl_noise, window_size)
    X_data_class_2_domain_2 = create_curve(frequencies[3], amplitudes[3], freq_noise, ampl_noise, window_size)
    
    
    for i in range(n-1):
        X_data_class_1_domain_1 = np.concatenate((X_data_class_1_domain_1, create_curve(frequencies[0], amplitudes[0], freq_noise, ampl_noise, window_size)), axis = 0) 
        X_data_class_2_domain_1 = np.concatenate((X_data_class_2_domain_1, create_curve(frequencies[1], amplitudes[1], freq_noise, ampl_noise, window_size)), axis = 0)
        X_data_class_1_domain_2 = np.concatenate((X_data_class_1_domain_2, create_curve(frequencies[2], amplitudes[2], freq_noise, ampl_noise, window_size)), axis = 0)
        X_data_class_2_domain_2 = np.concatenate((X_data_class_2_domain_2, create_curve(frequencies[3], amplitudes[3], freq_noise, ampl_noise, window_size)), axis = 0)
        #print(i)
        #print(np.shape(X_data_class_1_domain_1))
        if i%50==0:
            print(f"data loaded: {i}/{n}")
        
    n_samples = np.shape(X_data_class_2_domain_1)[0]*2  
    
    y_data_class_1_domain_1 = np.asarray([0]*np.shape(X_data_class_1_domain_1)[0])
    y_data_class_2_domain_1 = np.asarray([1]*np.shape(X_data_class_2_domain_1)[0])
    y_data_class_1_domain_2 = np.asarray([0]*np.shape(X_data_class_1_domain_2)[0])
    y_data_class_2_domain_2 = np.asarray([1]*np.shape(X_data_class_2_domain_2)[0])
    
    X_data_source = np.concatenate((X_data_class_1_domain_1, X_data_class_2_domain_1), axis = 0)
    y_data_source = np.concatenate((y_data_class_1_domain_1, y_data_class_2_domain_1), axis = 0)
    X_data_target = np.concatenate((X_data_class_1_domain_2, X_data_class_2_domain_2), axis = 0)
    y_data_target = np.concatenate((y_data_class_1_domain_2, y_data_class_2_domain_2), axis = 0)
    
    
    X_data_source = torch.from_numpy(X_data_source)
    y_data_source = torch.from_numpy(y_data_source)
    X_data_target = torch.from_numpy(X_data_target)
    y_data_target = torch.from_numpy(y_data_target)
    
    return n_samples, X_data_source, y_data_source, X_data_target, y_data_target


class Dataset_Dummy_Source_Window(Dataset):

    def __init__(self):

        n = 5000 #8000
        frequencies = [1,4,1.9,3.1]#[1,4,1.6,3.4]#[1,4,1.9,3.1]
        amplitudes = [6,2,5,4]
        freq_noise = 0.5 #0.3
        ampl_noise = 4 #2
        window_size = 1000
        self.n_samples, self.x_data, self.y_data, _, _ = create_data_window(n, frequencies, amplitudes, freq_noise, ampl_noise, window_size)
        
        
                  
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class Dataset_Dummy_Target_Window(Dataset):

    
    
    def __init__(self):

        n = 5000 #8000 
        frequencies = [1,4,1.9,3.1]#[1,4,1.6,3.4]#[1,4,1.9,3.1]
        amplitudes = [6,2,5,4]
        freq_noise = 0.5 #0.3
        ampl_noise = 4 #2
        window_size = 1000
        self.n_samples, _, _, self.x_data, self.y_data = create_data_window(n, frequencies, amplitudes, freq_noise, ampl_noise, window_size)
        
        
                  
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples




class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        
        """
        formula [(W−K+2P)/S]+1.
        """
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=100, stride=1)#input: 1000
        self.conv2 = nn.Conv1d(64,32,kernel_size=10, stride = 1, padding=1)#input: [(1000-100+2*0)/1]+1 = 901
        self.batch1 =nn.BatchNorm1d(32)#input: [(901-10+2*1)/1]+1 = 894
        self.conv3 = nn.Conv1d(32,32,kernel_size=5, stride = 1, padding=1) #input:894
        self.batch2 =nn.BatchNorm1d(32)#input: [(894-5+2*1)/1]+1 = 892
        #self.fc1 = nn.Linear(32*892, output_size)

    def forward(self, x):
        x = F.selu(self.conv1(x)) #conv1
        x = self.conv2(x) #conv2
        x = F.selu(self.batch1(x)) #batch1
        x = self.conv3(x) #conv3
        x = F.selu(self.batch2(x)) #batch2
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])) #flatten
        #x = self.fc1(x) #linear1
        output = x
        
        return output


#from mmd_loss import MMD_loss
class MMD_loss(nn.Module):
    def __init__(self, fix_sigma = None, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if torch.is_tensor(fix_sigma):
            bandwidth_list = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):

        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss



def forward(model, classifier_layer_1, classifier_layer_2, classifier_layer_3, data, labels_source, labels_target, criterion, MMD_loss_calculator, GAMMA):
        

        outputs = model(data.float())
        batch_size = len(labels_source)
        # cross-entropy loss
        

        out_src = outputs[:batch_size, :]
        out_tar = outputs[batch_size:, :]
        x_src = classifier_layer_1(outputs[:batch_size, :])
        x_tar = classifier_layer_1(outputs[batch_size:, :])
        source_out = classifier_layer_2(x_src)
        target_out = classifier_layer_2(x_tar)
        source_pred = classifier_layer_3(source_out)
        target_pred = classifier_layer_3(target_out)
        
        #CE loss
        ce_loss = criterion(source_pred, labels_source)
        target_ce_loss = criterion(target_pred, labels_target)

        # MMD Loss FC
        mmd_loss_1 = MMD_loss_calculator.forward(out_src, out_tar)
        mmd_loss_2 = MMD_loss_calculator.forward(x_src, x_tar)
        mmd_loss_3 = MMD_loss_calculator.forward(source_out, target_out)
        
        # MMD Loss Final
        mmd_loss =  mmd_loss_1 + mmd_loss_2 + mmd_loss_3
        #mmd_loss = mmd_loss_2
        #total loss
        loss = ce_loss + GAMMA * mmd_loss
        

        #collect information about labels, predictions
        n_correct_source = 0
        n_correct_target = 0
        n_samples_source = 0
        n_samples_target = 0
        
        
        # plot list
        class_0_source = np.empty((0,3))
        class_1_source = np.empty((0,3))
        class_0_target = np.empty((0,3))
        class_1_target = np.empty((0,3))
        
        
        #for each element in batch check if prediction is correct and collect total and correct predictions and labels
        for i in range(len(labels_source)):
            label_source = labels_source[i]
            output_source = torch.argmax(source_pred[i])
            if label_source == output_source:
                n_correct_source+=1
            n_samples_source+=1
            
            if label_source == 0:
                class_0_source = np.append(class_0_source, np.expand_dims(source_out[i,:].detach().numpy(), axis = 0), axis = 0)
            elif label_source == 1:
                class_1_source = np.append(class_1_source, np.expand_dims(source_out[i,:].detach().numpy(), axis = 0), axis = 0)
                
        acc_total_source = 100.0 * n_correct_source / n_samples_source
            
        for i in range(len(labels_target)):
            label_target = labels_target[i]
            output_target = torch.argmax(target_pred[i])
            if label_target == output_target:
                n_correct_target+=1
            n_samples_target+=1
                
            if label_target == 0:
                class_0_target = np.append(class_0_target, np.expand_dims(target_out[i,:].detach().numpy(), axis = 0), axis = 0)
            elif label_target == 1:
                class_1_target = np.append(class_1_target, np.expand_dims(target_out[i,:].detach().numpy(), axis = 0), axis = 0)

            

        acc_total_target = 100.0 * n_correct_target / n_samples_target

        
        return loss, mmd_loss, ce_loss, target_ce_loss, acc_total_source, acc_total_target, class_0_source, class_1_source, class_0_target, class_1_target
    
if __name__ == "__main__":
    #Tensorboard
    writer_graph = SummaryWriter('runs/Dataloader2/graph')
    writer_train = SummaryWriter('runs/Dataloader2/train')
    writer_val = SummaryWriter('runs/Dataloader2/val')
    writer_test = SummaryWriter('runs/Dataloader2/test')
    writer = {}
    writer["train"] = writer_train
    writer["val"] = writer_val
    writer["test"] = writer_test



    #Generate folder structure to store plots and data
    current_directory = os.getcwd()
    path_learning_curve = os.path.join(current_directory, "learning_curve")
    path_learning_curve_data = os.path.join(current_directory, "learning_curve_data")
    path_data_distribution = os.path.join(current_directory, "data_distribution")
    path_data_distribution_data = os.path.join(current_directory, "data_distribution_data")

    if not os.path.exists(path_learning_curve): #Folder to store Learning Curve Plots 
        os.makedirs(path_learning_curve)
    if not os.path.exists(path_learning_curve_data): #Folder to store Learning Curve Plots Data
        os.makedirs(path_learning_curve_data)
    if not os.path.exists(path_data_distribution): #Folder to store Data Distribuiton Plots 
        os.makedirs(path_data_distribution)
    if not os.path.exists(path_data_distribution_data): #Folder to store Data Distribuiton Plots Data 
        os.makedirs(path_data_distribution_data)

    # create csv file to store data 
    f_learning_curve = open(f'learning_curve_data/learning_curve.csv', 'w')

    # create csv writer to store data
    f_learning_curve_writer = csv.writer(f_learning_curve)
    f_learning_curve_writer.writerow(['loss_val', 'mmd_loss_val', 'source_ce_loss_val', 'target_ce_loss_val', 'acc_total_source_val', 'acc_total_target_val', 'loss_train', 'mmd_loss_train', 'source_ce_loss_train', 'target_ce_loss_train', 'acc_total_source_train', 'acc_total_target_train'])


    #collect loss for each batch
    loss_collected = 0
    target_ce_loss_collected = 0
    mmd_loss_collected = 0

    loss_list = {}
    loss_list['train']=[]
    loss_list['val']=[]

    target_ce_loss_list = {}
    target_ce_loss_list['train']=[]
    target_ce_loss_list['val']=[]

    source_ce_loss_list = {}
    source_ce_loss_list['train']=[]
    source_ce_loss_list['val']=[]

    mmd_loss_list = {}
    mmd_loss_list['train']=[]
    mmd_loss_list['val']=[]

    source_ce_loss_list = {}
    source_ce_loss_list['train']=[]
    source_ce_loss_list['val']=[]

    source_accuracy_list = {}
    source_accuracy_list['train']=[]
    source_accuracy_list['val']=[]

    target_accuracy_list = {}
    target_accuracy_list['train']=[]
    target_accuracy_list['val']=[]

    # check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"the device for executing the code is: {device}")


    #dataset source
    dataset_source = Dataset_Dummy_Source_Window()

    # define train/val dimensions source
    train_size_source = int(0.8 * len(dataset_source))
    validation_size_source = len(dataset_source) - train_size_source

    #split dataset source
    training_dataset_source, validation_dataset_source = torch.utils.data.random_split(dataset_source, [train_size_source, validation_size_source])

    batch_size = 64

    #dataloader source
    train_loader_source = DataLoader(dataset=training_dataset_source,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    validation_loader_source = DataLoader(dataset=validation_dataset_source,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)


    #dataset target
    dataset_target = Dataset_Dummy_Target_Window()

    # define train/val dimensions target
    train_size_target = int(0.8 * len(dataset_target))
    validation_size_target = len(dataset_target) - train_size_target

    #split dataset target
    training_dataset_target, validation_dataset_target = torch.utils.data.random_split(dataset_target, [train_size_target, validation_size_target])
    batch_size = 64

    #dataloader target
    train_loader_target = DataLoader(dataset=training_dataset_target,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    validation_loader_target = DataLoader(dataset=validation_dataset_target,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)

    #dataloader source
    source_loader = {}
    source_loader["train"] = train_loader_source
    source_loader["val"] = validation_loader_source

    #dataloader target
    target_loader = {}
    target_loader["test"] = train_loader_target
    target_loader["val"] = validation_loader_target


    #dataloader for training 
    dataloader_train = {}
    dataloader_train["source"]=source_loader["train"]
    dataloader_train["target"]=target_loader["test"]

    #dataloader for testing
    dataloader_val = {}
    dataloader_val["source"]=source_loader["val"]
    dataloader_val["target"]=target_loader["val"]

    #totdal datalaoder
    dataloaders = {}
    dataloaders["train"] = dataloader_train
    dataloaders["val"] = dataloader_val


    #define training params
    num_epochs = 10
    learning_rate = 0.008#0.008
    GAMMA = 20# 1000 more weight to transferability
    #SIGMA = [2,4,8]  # default 1
    SIGMA = torch.tensor([1,2,4,8,16],dtype=torch.float64)

    #models
    input_size = 1
    input_fc_size = 32*892 #25*40 
    hidden_fc_size_1 = 100
    hidden_fc_size_2 = 3
    output_size = 2


    model = CNN(input_size, output_size)

    classifier_layer_1 = nn.Linear(input_fc_size, hidden_fc_size_1)
    classifier_layer_2 = nn.Linear(hidden_fc_size_1, hidden_fc_size_2)
    classifier_layer_3 = nn.Linear(hidden_fc_size_2, output_size)

    #models to gpu if available
    model = model.to(device)
    classifier_layer_1 = classifier_layer_1.to(device)
    classifier_layer_2 = classifier_layer_2.to(device)
    classifier_layer_3 = classifier_layer_3.to(device)


    #define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    MMD_loss_calculator = MMD_loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    phases = ["val", "train"]
    # Train and Validate the model
    for epoch in range(num_epochs):
        print(f"epoch{epoch}/{num_epochs}")
        #plot mmd
        class_0_source_collect = np.empty((0,3))
        class_1_source_collect = np.empty((0,3))
        class_0_target_collect = np.empty((0,3))
        class_1_target_collect = np.empty((0,3))
        
        loss_collected = 0
        source_ce_loss_collected = 0
        target_ce_loss_collected = 0
        mmd_loss_collected = 0
        acc_total_source_collected = 0
        acc_total_target_collected = 0
        

        learning_curve_data_collect = []

        for phase in phases:
            iter_loader_source = iter(dataloaders[phase]["source"])
            iter_loader_target = iter(dataloaders[phase]["target"])
            
            for i in range(len(dataloaders[phase]["source"])):

                ########Forward pass########
                data_source, labels_source = iter_loader_source.next() #batch_size number of windows and labels from source domain
                data_target, labels_target = iter_loader_target.next() #batch_size number of windows from target domain
                data = torch.cat((data_source, data_target), dim=0) #concat the windows to 2*batch_size number of windows

                data=data.to(device)

                batch_size = len(labels_source) #take length of shorter dataoader which is the one from source domain (reason:train, val split)
                            
                if phase == "val":
                    
                    # no training
                    model.train(False)
                    classifier_layer_1.train(False)
                    classifier_layer_2.train(False)
                    classifier_layer_3.train(False)
                    
                    with torch.no_grad():
                        loss, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, class_0_source, class_1_source, class_0_target, class_1_target = forward(model, classifier_layer_1, classifier_layer_2, classifier_layer_3, data, labels_source, labels_target, criterion, MMD_loss_calculator, GAMMA)
                        
                        loss_collected += loss
                        mmd_loss_collected += mmd_loss
                        source_ce_loss_collected += source_ce_loss
                        target_ce_loss_collected += target_ce_loss
                        acc_total_source_collected += acc_total_source
                        acc_total_target_collected += acc_total_target
                        
                        # collect plot values
                        class_0_source_collect = np.append(class_0_source_collect, class_0_source, axis = 0)
                        class_1_source_collect = np.append(class_1_source_collect, class_1_source, axis = 0)
                        class_0_target_collect = np.append(class_0_target_collect, class_0_target, axis = 0)
                        class_1_target_collect = np.append(class_1_target_collect, class_1_target, axis = 0)
                        
                        
                
                elif phase == "train":
                    
                    # training
                    model.train(True)
                    classifier_layer_1.train(True)
                    classifier_layer_2.train(True)
                    classifier_layer_3.train(True)
                    
                                    
                    ########Forward pass########
                    loss, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, _, _, _, _ = forward(model, classifier_layer_1, classifier_layer_2, classifier_layer_3, data, labels_source, labels_target, criterion, MMD_loss_calculator, GAMMA)
                    
                    loss_collected += loss.detach()
                    mmd_loss_collected += mmd_loss.detach()
                    source_ce_loss_collected += source_ce_loss.detach()
                    target_ce_loss_collected += target_ce_loss.detach()
                    acc_total_source_collected += acc_total_source
                    acc_total_target_collected += acc_total_target

                    ########Backward pass########
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    
            #plot
            if phase == "val" and (epoch ==0 or epoch ==2 or epoch == 4 or epoch ==6 or epoch ==8 or epoch ==10):
                """
                fig = plt.figure()
                plt.gcf().set_size_inches((20, 20)) 
                ax = fig.add_subplot(projection='3d')

                m = [1,2,3,4]
                data = [class_0_source_collect, class_1_source_collect, class_0_target_collect, class_1_target_collect]
                data_label = ["Source: Class 0", "Source: Class 1", "Target: Class 0", "Target: Class 1"]
                for i in range(4):
                    ax.scatter(data[i][:,0], data[i][:,1], data[i][:,2], marker=m[i], label=data_label[i])
                
        
                
                #ax.legend(prop={'size': 13})
                
                
                ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=10, size=20)
                ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=10, size=20)
                ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=10, size=20)
                
                
                
                plt.rcParams.update({'font.size': 10})
                
                #plt.show()
                fig.savefig(f"data_distribution/no_mmd_epoch{epoch}", format='pdf')
                """
                df1 = pd.DataFrame({'class_0_source_fc2_collect_0_dim':class_0_source_collect[:, 0]})
                df2 = pd.DataFrame({'class_0_source_fc2_collect_1_dim':class_0_source_collect[:, 1]})
                df3 = pd.DataFrame({'class_0_source_fc2_collect_2_dim':class_0_source_collect[:, 2]})
                df4 = pd.DataFrame({'class_1_source_fc2_collect_0_dim':class_1_source_collect[:, 0]})
                df5 = pd.DataFrame({'class_1_source_fc2_collect_1_dim':class_1_source_collect[:, 1]})
                df6 = pd.DataFrame({'class_1_source_fc2_collect_2_dim':class_1_source_collect[:, 2]})
                df7 = pd.DataFrame({'class_0_target_fc2_collect_0_dim':class_0_target_collect[:, 0]})
                df8 = pd.DataFrame({'class_0_target_fc2_collect_1_dim':class_0_target_collect[:, 1]})
                df9 = pd.DataFrame({'class_0_target_fc2_collect_2_dim':class_0_target_collect[:, 2]})
                df10 = pd.DataFrame({'class_1_target_fc2_collect_0_dim':class_1_target_collect[:, 0]})
                df11 = pd.DataFrame({'class_1_target_fc2_collect_1_dim':class_1_target_collect[:, 1]})
                df12 = pd.DataFrame({'class_1_target_fc2_collect_2_dim':class_1_target_collect[:, 2]})
                pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12],axis=1).to_csv(f'data_distribution_data/data_distribution_{epoch}.csv', index = False)

                    
                    
        
            running_loss = loss_collected / len(dataloaders[phase]["source"])
            
            running_source_ce_loss = source_ce_loss_collected / len(dataloaders[phase]["source"])
            running_acc_source = acc_total_source_collected / len(dataloaders[phase]["source"])
            
            running_target_ce_loss = target_ce_loss_collected / len(dataloaders[phase]["source"])
            running_acc_target = acc_total_target_collected / len(dataloaders[phase]["source"])
            
            running_mmd_loss = mmd_loss_collected/ len(dataloaders[phase]["source"])

            
            learning_curve_data_collect = learning_curve_data_collect + [running_loss.item(), running_mmd_loss.item(), running_source_ce_loss.item(), running_target_ce_loss.item(), running_acc_source, running_acc_target]

            loss_collected = 0
            source_ce_loss_collected = 0
            target_ce_loss_collected = 0
            mmd_loss_collected = 0
            acc_total_source_collected = 0
            acc_total_target_collected = 0
            
            
            loss_list[phase].append(running_loss.item())

            target_ce_loss_list[phase].append(running_source_ce_loss.item())

            source_ce_loss_list[phase].append(running_target_ce_loss.item())

            mmd_loss_list[phase].append(running_mmd_loss.item())

            source_accuracy_list[phase].append(running_acc_source)
            
            target_accuracy_list[phase].append(running_acc_target)
            
            writer[phase].add_scalar(f'loss_list', running_loss.item(), epoch)
            writer[phase].add_scalar(f'target_ce_loss_list', running_source_ce_loss.item(), epoch)
            writer[phase].add_scalar(f'source_ce_loss_list', running_target_ce_loss.item(), epoch)
            writer[phase].add_scalar(f'mmd_loss_list', running_mmd_loss.item(), epoch)
            writer[phase].add_scalar(f'source_accuracy_list', running_acc_source, epoch)
            writer[phase].add_scalar(f'target_accuracy_list', running_acc_target, epoch)

            
            
        f_learning_curve_writer.writerow(learning_curve_data_collect)

        print(f"Epoch {epoch+1}/{num_epochs} successfull")
    
    f_learning_curve.close()



    """


    import pandas as pd

    fig1 = plt.figure()
    plt.title('Total Loss')
    plt.plot(loss_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(loss_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("Total Loss $\longrightarrow$")
    plt.show()
    fig1.savefig('learning_curve/total_loss', format='pdf')
    pd.DataFrame(loss_list['train']).to_csv('learning_curve_data/loss_list_train.csv',index=False,header=False)
    pd.DataFrame(loss_list['val']).to_csv('learning_curve_data/loss_list_val.csv',index=False,header=False)


    fig2 = plt.figure()
    plt.title('CE-Loss Target Domain')
    plt.plot(target_ce_loss_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(target_ce_loss_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("Target CE-Loss $\longrightarrow$")
    plt.show()
    fig2.savefig('learning_curve/target_ce_loss', format='pdf')
    pd.DataFrame(target_ce_loss_list['train']).to_csv('learning_curve_data/target_ce_loss_train.csv',index=False,header=False)
    pd.DataFrame(target_ce_loss_list['val']).to_csv('learning_curve_data/target_ce_loss_val.csv',index=False,header=False)

    fig3 = plt.figure()
    plt.title('CE-Loss Source Domain')
    plt.plot(source_ce_loss_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(source_ce_loss_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("Source CE-Loss $\longrightarrow$")
    plt.show()
    fig3.savefig('learning_curve/source_ce_loss', format='pdf')
    pd.DataFrame(source_ce_loss_list['train']).to_csv('learning_curve_data/source_ce_loss_train.csv',index=False,header=False)
    pd.DataFrame(source_ce_loss_list['val']).to_csv('learning_curve_data/source_ce_loss_val.csv',index=False,header=False)


    fig4 = plt.figure()
    plt.title('MMD-Loss')
    plt.plot(mmd_loss_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(mmd_loss_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("MMD-Loss $\longrightarrow$")
    plt.show()
    fig4.savefig('learning_curve/mmd_loss', format='pdf')
    pd.DataFrame(mmd_loss_list['train']).to_csv('learning_curve_data/mmd_loss_train.csv',index=False,header=False)
    pd.DataFrame(mmd_loss_list['val']).to_csv('learning_curve_data/mmd_loss_val.csv',index=False,header=False)




    fig5 = plt.figure()
    plt.title('Accuracy Source Domain')
    plt.plot(source_accuracy_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(source_accuracy_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("Source Accuracy $\longrightarrow$")
    plt.show()
    fig5.savefig('learning_curve/source_accuracy', format='pdf')
    pd.DataFrame(source_accuracy_list['train']).to_csv('learning_curve_data/source_accuracy_train.csv',index=False,header=False)
    pd.DataFrame(source_accuracy_list['val']).to_csv('learning_curve_data/source_accuracy_val.csv',index=False,header=False)



    fig6 = plt.figure()
    plt.title('Accuracy Target Domain')
    plt.plot(target_accuracy_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(target_accuracy_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("Target Accuracy $\longrightarrow$")
    plt.show()
    fig6.savefig('learning_curve/target_accuracy', format='pdf')
    pd.DataFrame(target_accuracy_list['train']).to_csv('learning_curve_data/target_accuracy_train.csv',index=False,header=False)
    pd.DataFrame(target_accuracy_list['val']).to_csv('learning_curve_data/target_accuracy_val.csv',index=False,header=False)

    """