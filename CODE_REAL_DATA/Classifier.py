import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, hidden_fc_size_1, hidden_fc_size_2, output_size, random_seed):
        """
        Constructor
        
        INPUT: 
        @hidden_fc_size_1: Dimension of FC1
        @hidden_fc_size_2: Dimension of FC2
        @output_size: Dimension of FC3
        @random_seed: Random Seed for weight init
        """
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(hidden_fc_size_1, hidden_fc_size_2)
        self.fc3 = nn.Linear(hidden_fc_size_2, output_size)
        torch.manual_seed(random_seed)

    def forward(self, x):
        """
        Forward Pass Classifier

        INPUT:
        @x: Input in CNN

        OUTPUT:
        @x_fc2: : Latent Feature Representation of Fully Connected Layer 2
        @x_fc3: : Latent Feature Representation of Fully Connected Layer 3
        """
        x_fc2 = self.fc2(x) #fc2
        x_fc3 = self.fc3(x_fc2) #fc3
        
        return x_fc2, x_fc3