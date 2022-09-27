import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np



def plot_distribution():
        
    #get all data_distribution files recorded during training
    csv_file_path_folder = os.path.join("data_distribution_data")
    csv_file_path_folder_elements = os.listdir(csv_file_path_folder)


    #for each of those files generate a figure
    for element in csv_file_path_folder_elements:
            
        #read csv file as pandas
        csv_file_path = os.path.join("data_distribution_data", element)
        df = pd.read_csv(csv_file_path)
        #create figure
        fig = plt.figure()
        plt.gcf().set_size_inches((20, 20)) 
        ax = fig.add_subplot(projection='3d')

        #plot
        m_colour = ['r','g','b','y'] #colours
        m_form = ["o","o","o","o"] #form
        columns = df.columns.values.tolist()
        for i in range(4):
            ax.scatter(df[columns[0+i*3]][::1], df[columns[1+i*3]][::1], df[columns[2+i*3]][::1], marker=m_form[i], c=m_colour[i],  s=80)
            
        #label axis
        ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=30, size=30)
        ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=30, size=30)
        ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=30, size=30)

        ax.tick_params(axis="x", pad=10, labelsize= 18)
        ax.tick_params(axis="y", pad=10, labelsize= 18)
        ax.tick_params(axis="z", pad=10, labelsize= 18)
        

        #set fig size
        plt.rcParams.update({'font.size': 10})

        #safe figure 
        fig.savefig(f"data_distribution/{element[:-4]}.pdf", format='pdf')     
def plot_curves():
        
        
    #read csv file as pandas
    csv_file_path = os.path.join("learning_curve_data", "learning_curve.csv")
    df = pd.read_csv(f'{csv_file_path}')

    #Plot Accuracy Source
    fig1 = plt.figure()
    plt.title('Accuracy Source Domain', fontsize=15)
    plt.plot(df["acc_total_source_train"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(df["acc_total_source_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=15)
    plt.ylabel("Accuracy Source Domain $\longrightarrow$", fontsize=15)
    plt.legend(prop={'size': 13})
    fig1.savefig(f"learning_curve/Accuracy_Source_Domain.pdf", format='pdf')

    #Plot Accuracy Target
    fig2 = plt.figure()
    plt.title('Accuracy Target Domain', fontsize=15)
    plt.plot(df["acc_total_target_train"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(df["acc_total_target_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=15)
    plt.ylabel("Accuracy Target Domain $\longrightarrow$", fontsize=15)
    plt.legend(prop={'size': 13})
    fig2.savefig(f"learning_curve/Accuracy_Target_Domain.pdf", format='pdf')

    #Plot CE Loss Source
    fig3 = plt.figure()
    plt.title('CE-Loss Source Domain', fontsize=15)
    plt.plot(df["source_ce_loss_train"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(df["source_ce_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=15)
    plt.ylabel("CE-Loss Source Domain $\longrightarrow$", fontsize=15)
    plt.legend(prop={'size': 13})
    fig3.savefig(f"learning_curve/CE_Loss_Source_Domain.pdf", format='pdf')

    #Plot CE Loss Target
    fig4 = plt.figure()
    plt.title('CE-Loss Target Domain', fontsize=15)
    plt.plot(df["target_ce_loss_train"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(df["target_ce_loss_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=15)
    plt.ylabel("CE-Loss Target Domain $\longrightarrow$", fontsize=15)
    plt.legend(prop={'size': 13})
    fig4.savefig(f"learning_curve/CE_Loss_Target_Domain.pdf", format='pdf')

    #Plot MMD Loss
    fig5 = plt.figure()
    plt.title('MMD-Loss', fontsize=15)
    plt.plot(df["mmd_loss_train"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(df["mmd_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=15)
    plt.ylabel("MMD-Loss $\longrightarrow$", fontsize=1150)
    plt.legend(prop={'size': 13})
    fig5.savefig(f"learning_curve/MMD_Loss.pdf", format='pdf')

plot_distribution()
plot_curves()

