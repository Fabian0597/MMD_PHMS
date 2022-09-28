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
            m_colour = ['r','g','b','y'] #colours
            m_form = ["o","o","o","o"] #form
            m_legend = ["Source: Class 0", "Source: Class 1", "Target: Class 0", "Target: Class 1"]
            columns = df.columns.values.tolist()
            for i in range(4):
                ax.scatter(df[columns[0+i*3]][::2], df[columns[1+i*3]][::2], df[columns[2+i*3]][::2], marker=m_form[i], c=m_colour[i],  s=100, label=m_legend[i])
        
            #label axis
            ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=70, size=45)
            ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=70, size=45)
            ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=90, size=45)

            x_ticks = ax.get_xticks()
            x_ticks = x_ticks[0::2]
            #x_ticks = x_ticks[:-1]

            y_ticks = ax.get_yticks()
            y_ticks = y_ticks[0::2]
            y_ticks = y_ticks[1:]

            z_ticks = ax.get_zticks()
            z_ticks = z_ticks[0::2]
            z_ticks = z_ticks[1:]

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_zticks(z_ticks)


            ax.tick_params(axis="x", pad=20, labelsize= 27)
            ax.tick_params(axis="y", pad=20, labelsize= 27)
            ax.tick_params(axis="z", pad=40, labelsize= 27)


            #set fig size
            plt.rcParams.update({'font.size': 10})
            ax.legend(prop={'size': 32}, markerscale=4, loc='center left')

            #safe figure 
            fig.savefig(f"{self.datapath}/data_distribution/{element[:-4]}.pdf", format='pdf')
            plt.close(fig)


    def plot_curves(self):
        
        #read csv file as pandas
        csv_file_path = os.path.join(self.datapath, "learning_curve_data", "learning_curve.csv")
        df = pd.read_csv(f'{csv_file_path}')

        #Plot Accuracy Source
        fig1 = plt.figure()
        plt.title('Accuracy Source Domain', fontsize=22)
        plt.plot(df["running_acc_source_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_source_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_source_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=22)
        plt.ylabel("Accuracy Source Domain $\longrightarrow$", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=17)
        plt.tight_layout()
        fig1.savefig(f"{self.datapath}/learning_curve/Accuracy_Source_Domain.pdf", format='pdf')
        plt.close(fig1)

        #Plot Accuracy Target
        fig2 = plt.figure()
        plt.title('Accuracy Target Domain', fontsize=22)
        plt.plot(df["running_acc_target_ce"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_target_mmd"], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_target_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=22)
        plt.ylabel("Accuracy Target Domain $\longrightarrow$", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=17)
        plt.tight_layout()
        fig2.savefig(f"{self.datapath}/learning_curve/Accuracy_Target_Domain.pdf", format='pdf')
        plt.close(fig2)

        #Plot CE Loss Source
        fig3 = plt.figure()
        plt.title('CE-Loss Source Domain', fontsize=22)
        plt.plot(df["running_source_ce_loss_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_source_ce_loss_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_source_ce_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=22)
        plt.ylabel("CE-Loss Source Domain $\longrightarrow$", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=17)
        plt.tight_layout()
        fig3.savefig(f"{self.datapath}/learning_curve/CE_Loss_Source_Domain.pdf", format='pdf')
        plt.close(fig3)
    
        #Plot CE Loss Target
        fig4 = plt.figure()
        plt.title('CE-Loss Target Domain', fontsize=22)
        plt.plot(df["running_target_ce_loss_ce"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_target_ce_loss_mmd"], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_target_ce_loss_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=22)
        plt.ylabel("CE-Loss Target Domain $\longrightarrow$", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=17)
        plt.tight_layout()
        fig4.savefig(f"{self.datapath}/learning_curve/CE_Loss_Target_Domain.pdf", format='pdf')
        plt.close(fig4)

        #Plot MMD Loss
        fig5 = plt.figure()
        plt.title('MMD-Loss', fontsize=22)
        plt.plot(df["running_mmd_loss_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_mmd_loss_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_mmd_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$", fontsize=22)
        plt.ylabel("MMD-Loss $\longrightarrow$", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=17)
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
            plotter.plot_distribution()
            plotter.plot_curves()

