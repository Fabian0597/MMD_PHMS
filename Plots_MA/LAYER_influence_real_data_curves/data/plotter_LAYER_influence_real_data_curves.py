import pandas as pd
import matplotlib.pyplot as plt
import os

class Plotter():
    def __init__(self, datapath):
        self.datapath = datapath

    def plot_distribution(self):
        
        #get all data_distribution files recorded during training
        csv_file_path_folder = os.path.join(self.datapath, "data_distribution_data")
        csv_file_path_folder_elements = os.listdir(csv_file_path_folder)

        #for each of those files generate a figure
        for element in csv_file_path_folder_elements:

            #read csv file as pandas
            csv_file_path = os.path.join(self.datapath, "data_distribution_data", element)
            df = pd.read_csv(csv_file_path)

            #create figure
            fig = plt.figure()
            plt.gcf().set_size_inches((20, 20)) 
            ax = fig.add_subplot(projection='3d')

            #plot
            m = [1,2,3,4] #colours
            columns = df.columns.values.tolist()
            for i in range(4):
                ax.scatter(df[columns[0+i*3]], df[columns[1+i*3]], df[columns[2+i*3]], marker=m[i])
        
            #label axis
            ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=50, size=45)
            ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=50, size=45)
            ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=50, size=45)

            ax.tick_params(axis="x", pad=15, labelsize= 27)
            ax.tick_params(axis="y", pad=15, labelsize= 27)
            ax.tick_params(axis="z", pad=15, labelsize= 27)

            #set fig size
            plt.rcParams.update({'font.size': 10})

            #safe figure 
            fig.savefig(f"{self.datapath}/data_distribution/{element[:-4]}.pdf", format='pdf')
            plt.close(fig)


    def plot_curves(self):
        
        #read csv file as pandas
        csv_file_path = os.path.join(self.datapath, "learning_curve_data", "learning_curve.csv")
        df = pd.read_csv(f'{csv_file_path}')

        #Plot Accuracy Source
        fig1 = plt.figure()
        plt.title('Accuracy Source Domain', fontsize=21) #max 22
        plt.plot(df["running_acc_source_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_source_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_source_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=19) #max 22
        plt.ylabel("Accuracy Source Domain (%) $\longrightarrow$", fontsize=19) #max 22
        plt.xticks(fontsize=17) #max 19
        plt.yticks(fontsize=17) #max 19
        plt.legend(fontsize=15) #max 17
        plt.tight_layout()
        fig1.savefig(f"{self.datapath}/learning_curve/Accuracy_Source_Domain.pdf", format='pdf')
        plt.close(fig1)

        #Plot Accuracy Target
        fig2 = plt.figure()
        plt.title('Accuracy Target Domain', fontsize=21)
        plt.plot(df["running_acc_target_ce"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_target_mmd"], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_target_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=19)
        plt.ylabel("Accuracy Target Domain (%) $\longrightarrow$", fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=15)
        plt.tight_layout()
        fig2.savefig(f"{self.datapath}/learning_curve/Accuracy_Target_Domain.pdf", format='pdf')
        plt.close(fig2)

        #Plot CE Loss Source
        fig3 = plt.figure()
        plt.title('CE-Loss Source Domain', fontsize=21)
        plt.plot(df["running_source_ce_loss_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_source_ce_loss_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_source_ce_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=19)
        plt.ylabel("CE-Loss Source Domain $\longrightarrow$", fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=15)
        plt.tight_layout()
        fig3.savefig(f"{self.datapath}/learning_curve/CE_Loss_Source_Domain.pdf", format='pdf')
        plt.close(fig3)
    
        #Plot CE Loss Target
        fig4 = plt.figure()
        plt.title('CE-Loss Target Domain', fontsize=21)
        plt.plot(df["running_target_ce_loss_ce"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_target_ce_loss_mmd"], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_target_ce_loss_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=19)
        plt.ylabel("CE-Loss Target Domain $\longrightarrow$", fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=15)
        plt.tight_layout()
        fig4.savefig(f"{self.datapath}/learning_curve/CE_Loss_Target_Domain.pdf", format='pdf')
        plt.close(fig4)

        #Plot MMD Loss
        fig5 = plt.figure()
        plt.title('MMD-Loss', fontsize=21)
        plt.plot(df["running_mmd_loss_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_mmd_loss_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_mmd_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=19)
        plt.ylabel("MMD-Loss $\longrightarrow$", fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=15)
        plt.tight_layout()
        fig5.savefig(f"{self.datapath}/learning_curve/MMD_Loss.pdf", format='pdf')
        plt.close(fig5)

if __name__ == "__main__":

    cwd = os.getcwd()
    for folder in os.listdir(cwd):
        if not (folder.startswith(".") or folder.endswith(".py")):
            folder_path = os.path.join(cwd, folder)
            for file in os.listdir(folder_path):
                if not (file.startswith(".") or file.endswith(".py")):
                    file_path = os.path.join(folder_path, file)
                    print(file_path)
                    plotter = Plotter(file_path)
                    #plotter.plot_distribution()
                    plotter.plot_curves()

