import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import genfromtxt
import csv
import torch

class Plotter():
    def __init__(self, datapath):
        self.datapath = datapath

    def plot_distribution(self):
        csv_file_path_folder = os.path.join(self.datapath, "data_distribution_data")
        csv_file_path_folder_elements = os.listdir(csv_file_path_folder)

        class1 = np.empty((0,3), int)
        class2 = np.empty((0,3), int)
        class3 = np.empty((0,3), int)
        class4 = np.empty((0,3), int)
        classes = [class1, class2, class3, class4]
        print(class1)

        #for each of those files generate a figure
        for element in csv_file_path_folder_elements:
            print(element)
            with open(os.path.join(csv_file_path_folder, element), 'r') as csvfile:
                datareader = csv.reader(csvfile)
                i = 0
                for row in datareader:
                    for data_point in row:

                        data_point_1 = data_point.split('[')
                        data_point_2 = data_point_1[1].split(']')
                        data_point_3 = data_point_2[0].replace(" ", "")
                        data_point_4 = np.array(data_point_3.split(','))
                        classes[i] = np.vstack((classes[i], data_point_4))
                    classes[i] = np.array(classes[i],dtype=float)
                    print(np.shape(classes[i]))
                    i+=1
                    

            #create figure
            fig = plt.figure()
            plt.gcf().set_size_inches((20, 20)) 
            ax = fig.add_subplot(projection='3d')

            #plot
            m = [1,2,3,4] #colours
            for i in range(4):
                #ax.scatter(df[columns[0+i*3]], df[columns[1+i*3]], df[columns[2+i*3]], marker=m[i])
                ax.scatter(classes[i][:,0], classes[i][:,1], classes[i][:,2], marker=m[i])
        
            #label axis
            ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=40, size=35)
            ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=40, size=35)
            ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=50, size=35)

            ax.tick_params(axis="x", pad=10, labelsize= 23)
            ax.tick_params(axis="y", pad=10, labelsize= 23)
            ax.tick_params(axis="z", pad=20, labelsize= 23)


            #set fig size
            plt.rcParams.update({'font.size': 10})
            #safe figure 
            fig.savefig(f"{self.datapath}/data_distribution/{element[:-4]}.pdf", format='pdf')
            plt.close(fig)
            
            

    def plot_curves(self):
        

        accuracy_list_source_ce_path = os.path.join(self.datapath, "learning_curve_data", "accuracy_list_source_ce.csv")
        accuracy_list_source_mmd_path = os.path.join(self.datapath, "learning_curve_data", "accuracy_list_source_mmd.csv")
        accuracy_list_source_val_path = os.path.join(self.datapath, "learning_curve_data", "accuracy_list_source_val.csv")
        accuracy_list_target_ce_path = os.path.join(self.datapath, "learning_curve_data", "accuracy_list_target_ce.csv")
        accuracy_list_target_mmd_path = os.path.join(self.datapath, "learning_curve_data", "accuracy_list_target_mmd.csv")
        accuracy_list_target_val_path = os.path.join(self.datapath, "learning_curve_data", "accuracy_list_target_val.csv")
        ce_loss_list_source_ce_path = os.path.join(self.datapath, "learning_curve_data", "ce_loss_list_source_ce.csv")
        ce_loss_list_source_mmd_path = os.path.join(self.datapath, "learning_curve_data", "ce_loss_list_source_mmd.csv")
        ce_loss_list_source_val_path = os.path.join(self.datapath, "learning_curve_data", "ce_loss_list_source_val.csv")
        ce_loss_list_target_ce_path = os.path.join(self.datapath, "learning_curve_data", "ce_loss_list_target_ce.csv")
        ce_loss_list_target_mmd_path = os.path.join(self.datapath, "learning_curve_data", "ce_loss_list_target_mmd.csv")
        ce_loss_list_target_val_path = os.path.join(self.datapath, "learning_curve_data", "ce_loss_list_target_val.csv")
        mmd_loss_list_ce_path = os.path.join(self.datapath, "learning_curve_data", "mmd_loss_list_ce.csv")
        mmd_loss_list_mmd_path = os.path.join(self.datapath, "learning_curve_data", "mmd_loss_list_mmd.csv")
        mmd_loss_list_val_path = os.path.join(self.datapath, "learning_curve_data", "mmd_loss_list_val.csv")

        accuracy_list_source_ce = genfromtxt(accuracy_list_source_ce_path, delimiter='\n')
        accuracy_list_source_mmd = genfromtxt(accuracy_list_source_mmd_path, delimiter='\n')
        accuracy_list_source_val = genfromtxt(accuracy_list_source_val_path, delimiter='\n')
        accuracy_list_target_ce = genfromtxt(accuracy_list_target_ce_path, delimiter='\n')
        accuracy_list_target_mmd = genfromtxt(accuracy_list_target_mmd_path, delimiter='\n')
        accuracy_list_target_val = genfromtxt(accuracy_list_target_val_path, delimiter='\n')
        ce_loss_list_source_ce = genfromtxt(ce_loss_list_source_ce_path, delimiter='\n')
        ce_loss_list_source_mmd = genfromtxt(ce_loss_list_source_mmd_path, delimiter='\n')
        ce_loss_list_source_val = genfromtxt(ce_loss_list_source_val_path, delimiter='\n')
        ce_loss_list_target_ce = genfromtxt(ce_loss_list_target_ce_path, delimiter='\n')
        ce_loss_list_target_mmd = genfromtxt(ce_loss_list_target_mmd_path, delimiter='\n')
        ce_loss_list_target_val = genfromtxt(ce_loss_list_target_val_path, delimiter='\n')
        mmd_loss_list_ce = genfromtxt(mmd_loss_list_ce_path, delimiter='\n')
        mmd_loss_list_mmd = genfromtxt(mmd_loss_list_mmd_path, delimiter='\n')
        mmd_loss_list_val = genfromtxt(mmd_loss_list_val_path, delimiter='\n')
    

        #Plot Accuracy Source
        fig1 = plt.figure()
        plt.title('Accuracy Source Domain', fontsize=17)
        plt.plot(accuracy_list_source_ce, 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(accuracy_list_source_mmd, 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(accuracy_list_source_val, 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
        plt.ylabel("Accuracy Source Domain (%) $\longrightarrow$", fontsize=17)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        fig1.savefig(f"{self.datapath}/learning_curve/Accuracy_Source_Domain.pdf", format='pdf')
        plt.close(fig1)

        #Plot Accuracy Target
        fig2 = plt.figure()
        plt.title('Accuracy Target Domain', fontsize=17)
        plt.plot(accuracy_list_target_ce, 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(accuracy_list_target_mmd, 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(accuracy_list_target_val, 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
        plt.ylabel("Accuracy Target Domain (%) $\longrightarrow$", fontsize=17)
        plt.yticks(np.array([50, 55, 60, 65, 70, 75, 80, 85, 90]))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        fig2.savefig(f"{self.datapath}/learning_curve/Accuracy_Target_Domain.pdf", format='pdf')
        plt.close(fig2)

        #Plot CE Loss Source
        fig3 = plt.figure()
        plt.title('CE-Loss Source Domain', fontsize=17)
        plt.plot(ce_loss_list_source_ce, 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(ce_loss_list_source_mmd, 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(ce_loss_list_source_val, 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
        plt.ylabel("CE-Loss Source Domain $\longrightarrow$", fontsize=17)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        fig3.savefig(f"{self.datapath}/learning_curve/CE_Loss_Source_Domain.pdf", format='pdf')
        plt.close(fig3)
    
        #Plot CE Loss Target
        fig4 = plt.figure()
        plt.title('CE-Loss Target Domain', fontsize=17)
        plt.plot(ce_loss_list_target_ce, 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(ce_loss_list_target_mmd, 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(ce_loss_list_target_val, 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
        plt.ylabel("CE-Loss Target Domain $\longrightarrow$", fontsize=17)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        fig4.savefig(f"{self.datapath}/learning_curve/CE_Loss_Target_Domain.pdf", format='pdf')
        plt.close(fig4)

        #Plot MMD Loss
        fig5 = plt.figure()
        plt.title('MMD-Loss', fontsize=17)
        plt.plot(mmd_loss_list_ce, 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(mmd_loss_list_mmd, 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(mmd_loss_list_val, 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
        plt.ylabel("MMD-Loss $\longrightarrow$", fontsize=17)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        fig5.savefig(f"{self.datapath}/learning_curve/MMD_Loss.pdf", format='pdf')
        plt.close(fig5)
    

if __name__ == "__main__":

    cwd = os.getcwd()
    for file in os.listdir(cwd):
        if not (file.startswith(".") or file.endswith(".py")):
            file_path = os.path.join(cwd, file)
            print(file_path)
            plotter = Plotter(file_path)
            plotter.plot_curves()
            #plotter.plot_distribution()

