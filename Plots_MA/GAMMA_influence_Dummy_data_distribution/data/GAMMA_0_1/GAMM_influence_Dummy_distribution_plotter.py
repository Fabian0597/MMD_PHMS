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
        m_legend = ["Source: Class 0", "Source: Class 1", "Target: Class 0", "Target: Class 1"]
        columns = df.columns.values.tolist()
        for i in range(4):
            ax.scatter(df[columns[0+i*3]][::2], df[columns[1+i*3]][::2], df[columns[2+i*3]][::2], marker=m_form[i], c=m_colour[i],  s=100, label=m_legend[i])
        
        #label axis
        ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=50, size=45)
        ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=50, size=45)
        ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=50, size=45)

        ax.tick_params(axis="x", pad=20, labelsize= 27)
        ax.tick_params(axis="y", pad=20, labelsize= 27)
        ax.tick_params(axis="z", pad=20, labelsize= 27)

        x_ticks = ax.get_xticks()
        #ax.set_xticks([1,0.5, 0, -0.5, -1, -1.5, -2])
        x_ticks = x_ticks[0::2]
        #x_ticks = x_ticks[:-1]

        y_ticks = ax.get_yticks()
        y_ticks = y_ticks[0::2]
        #y_ticks = y_ticks[1:]

        z_ticks = ax.get_zticks()
        z_ticks = z_ticks[0::2]
        #z_ticks = z_ticks[1:]

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)

        legend = ax.legend(prop={'size': 32}, markerscale=4, loc='center left')
        
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
    plt.title('Accuracy Source Domain', fontsize=17)
    plt.plot(df["acc_total_source_train"], 'bo-', label = 'Train', linewidth=1,markersize=0.1)
    plt.plot(df["acc_total_source_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
    plt.ylabel("Accuracy Source Domain $\longrightarrow$", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    fig1.savefig(f"learning_curve/Accuracy_Source_Domain.pdf", format='pdf')

    #Plot Accuracy Target
    fig2 = plt.figure()
    plt.title('Accuracy Target Domain', fontsize=17)
    plt.plot(df["acc_total_target_train"], 'co-', label = 'Train', linewidth=1,markersize=0.1)
    plt.plot(df["acc_total_target_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
    plt.ylabel("Accuracy Target Domain $\longrightarrow$", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    fig2.savefig(f"learning_curve/Accuracy_Target_Domain.pdf", format='pdf')

    #Plot CE Loss Source
    fig3 = plt.figure()
    plt.title('CE-Loss Source Domain', fontsize=17)
    plt.plot(df["source_ce_loss_train"], 'bo-', label = 'Train', linewidth=1,markersize=0.1)
    plt.plot(df["source_ce_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
    plt.ylabel("CE-Loss Source Domain $\longrightarrow$", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    fig3.savefig(f"learning_curve/CE_Loss_Source_Domain.pdf", format='pdf')

    #Plot CE Loss Target
    fig4 = plt.figure()
    plt.title('CE-Loss Target Domain', fontsize=17)
    plt.plot(df["target_ce_loss_train"], 'co-', label = 'Train', linewidth=1,markersize=0.1)
    plt.plot(df["target_ce_loss_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
    plt.ylabel("CE-Loss Target Domain $\longrightarrow$", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    fig4.savefig(f"learning_curve/CE_Loss_Target_Domain.pdf", format='pdf')

    #Plot MMD Loss
    fig5 = plt.figure()
    plt.title('MMD-Loss', fontsize=17)
    plt.plot(df["mmd_loss_train"], 'bo-', label = 'Train', linewidth=1,markersize=0.1)
    plt.plot(df["mmd_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$", fontsize=17)
    plt.ylabel("MMD-Loss $\longrightarrow$", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    fig5.savefig(f"learning_curve/MMD_Loss.pdf", format='pdf')

plot_distribution()
#plot_curves()

