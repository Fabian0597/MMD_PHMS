import torch

class Loss_CNN():
    def __init__(self, model_cnn, model_fc, criterion, MMD_loss_calculator, MMD_loss_CNN_calculator, MMD_layer_activation_flag):
        """
        Constructor

        INPUT:
        @model_cnn: CNN model
        @model_fc: Classifier model
        @criterion: CE-Loss criterion
        @MMD_loss_calculator: MMD-Loss criterion of FC layers
        @MMD_loss_CNN_calculator: MMD-Loss criterion of CNN layers
        @MMD_layer_activation_flag: List of which Layers are used in the MMD-Loss
        """
        
        self.model_cnn = model_cnn
        self.model_fc = model_fc
        self.criterion = criterion
        self.MMD_loss_calculator = MMD_loss_calculator
        self.MMD_loss_CNN_calculator = MMD_loss_CNN_calculator
        self.MMD_layer_activation_flag = MMD_layer_activation_flag
 
    
    def forward(self, batch_data, labels_source, labels_target, mmd_loss_flag_phase, GAMMA):
        """
        Forward Pass: Calculation of Loss for Model Optimization

        INPUT:
        @batch_data: Data from the current batch in Training Loop
        @labels_source: Source Domain Labels
        @labels_target: Target Domain Labels
        @mmd_loss_flag_phase: Flag which indicates the MMD-Loss usage (Training Phase 1 = True, Trainign Phase 2 = False)
        @GAMMA: Hyperparameter to weight the the CE-Loss and MMD-Loss in Training Phase 1

        OUTPUT:
        @loss: Total Loss used for model optimization for Training Phase 1 and Training Phase 2
        @mmd_loss: MMD-Loss
        @source_ce_loss: CE-Loss on Source Domain for batch
        @target_ce_loss: CE-Loss on Target Domain for batch
        @acc_total_source: Accuracy on Source Domain for batch
        @acc_total_target: Accuracy on Target Domain for batch
        @balanced_target_accuracy: Balanced Accuracy on Target Domain for batch
        @class_0_source_fc2: List of Latent Space Representation of samples belonging to Source Domain and Class 0 in FC2 for data distribution plot
        @class_1_source_fc2: List of Latent Space Representation of samples belonging to Source Domain and Class 1 in FC2 for data distribution plot
        @class_0_target_fc2: List of Latent Space Representation of samples belonging to Target Domain and Class 0 in FC2 for data distribution plot
        @class_1_target_fc2: List of Latent Space Representation of samples belonging to Target Domain and Class 1 in FC2 for data distribution plot

        """

        #Forward Pass
        x_conv_1, x_conv_2, x_conv_3, x_flatten, x_fc1 = self.model_cnn(batch_data.float())
        x_fc2, x_fc3 = self.model_fc(x_fc1)
        
        #Retrieve Model Latent Feature Space Representations for MMD-Loss
        x_conv_1 = x_conv_1.to("cpu")
        x_conv_2 = x_conv_2.to("cpu")
        x_conv_3 = x_conv_3.to("cpu")
        x_flatten = x_flatten.to("cpu")
        x_fc1 = x_fc1.to("cpu")
        x_fc2 = x_fc2.to("cpu")
        x_fc3 = x_fc3.to("cpu")

        if x_conv_1.is_cuda:
            print("Latent output of model is also on GPU and might cause problems in criterion")

        batch_size = len(labels_source)   

        #CE Loss
        source_ce_loss = self.criterion(x_fc3[:batch_size, :], labels_source)
        target_ce_loss = self.criterion(x_fc3[batch_size:, :], labels_target)
        
        #MMD-Loss for CNN Layers
        if self.MMD_layer_activation_flag[0] == True:
            mmd_loss_1_cnn = self.MMD_loss_CNN_calculator.forward(x_conv_1[:batch_size, :, :], x_conv_1[batch_size:, :, :])
        else:
            mmd_loss_1_cnn = 0
        if self.MMD_layer_activation_flag[1] == True:
            mmd_loss_2_cnn = self.MMD_loss_CNN_calculator.forward(x_conv_2[:batch_size, :, :], x_conv_2[batch_size:,:, :])
        else:
            mmd_loss_2_cnn = 0
        if self.MMD_layer_activation_flag[2] == True:
            mmd_loss_3_cnn = self.MMD_loss_CNN_calculator.forward(x_conv_3[:batch_size, :, :], x_conv_3[batch_size:,:, :])
        else:
            mmd_loss_3_cnn = 0

        #MMD-Loss for FC Layers
        if self.MMD_layer_activation_flag[3] == True:
            mmd_loss_1_fc = self.MMD_loss_calculator.forward(x_flatten[:batch_size, :], x_flatten[batch_size:, :])
        else:
            mmd_loss_1_fc = 0
        if self.MMD_layer_activation_flag[4] == True:
            mmd_loss_2_fc = self.MMD_loss_calculator.forward(x_fc1[:batch_size, :], x_fc1[batch_size:, :])
        else:
            mmd_loss_2_fc = 0
        if self.MMD_layer_activation_flag[5] == True:
            mmd_loss_3_fc = self.MMD_loss_calculator.forward(x_fc2[:batch_size, :], x_fc2[batch_size:, :])
        else:
            mmd_loss_3_fc = 0
        
        #MMD Loss for CNN Layers (old and slow without Vector Matrix Calculation)
        #mmd_loss_1_cnn = 0
        #mmd_loss_2_cnn = 0
        #mmd_loss_3_cnn = 0
        #for channel1 in range(x_conv_1.size()[1]):
            #mmd_loss_1_cnn += self.MMD_loss_calculator.forward(x_conv_1[:batch_size, channel1, :], x_conv_1[batch_size:,channel1, :])
        #for channel2 in range(x_conv_2.size()[1]):
            #mmd_loss_2_cnn += self.MMD_loss_calculator.forward(x_conv_2[:batch_size, channel2, :], x_conv_2[batch_size:,channel2, :])
        #for channel3 in range(x_conv_3.size()[1]):
            #mmd_loss_3_cnn += self.MMD_loss_calculator.forward(x_conv_3[:batch_size, channel3, :], x_conv_3[batch_size:,channel3, :])

        #Total MMD Loss
        mmd_loss =  GAMMA * (mmd_loss_1_cnn + mmd_loss_2_cnn + mmd_loss_3_cnn + mmd_loss_1_fc + mmd_loss_2_fc + mmd_loss_3_fc)

        # list of latent space features in FC2 for data distribution plot
        class_0_source_fc2 = x_fc2[:batch_size, :][labels_source==0]
        class_1_source_fc2 = x_fc2[:batch_size, :][labels_source==1]
        class_0_target_fc2 = x_fc2[batch_size:, :][labels_target==0]
        class_1_target_fc2 = x_fc2[batch_size:, :][labels_target==1]
        
        #Accuracy Source Domain averaged over batch
        argmax_source_pred = torch.argmax(x_fc3[:batch_size, :], dim=1)
        result_source_pred = argmax_source_pred == labels_source
        correct_source_pred = result_source_pred[result_source_pred == True]
        acc_total_source = 100 * len(correct_source_pred)/len(labels_source)
        
        #Accuracy Target Domain averaged over batch
        argmax_target_pred = torch.argmax(x_fc3[batch_size:, :], dim=1)
        result_target_pred = argmax_target_pred == labels_target
        correct_target_pred = result_target_pred[result_target_pred == True]
        acc_total_target = 100 * len(correct_target_pred)/len(labels_target)

        #Get correct classified Target domain samples for each class
        result_target_pred_class_0 = result_target_pred[labels_target == 0]
        result_target_pred_class_1 = result_target_pred[labels_target == 1]
        correct_target_pred_class_0 = result_target_pred_class_0[result_target_pred_class_0 == True]
        correct_target_pred_class_1 = result_target_pred_class_1[result_target_pred_class_1 == True]

        #Specifity
        if len(labels_target[labels_target==0]) == 0:
            acc_total_target_class_0 = 0
        else:
            acc_total_target_class_0 = 100 * len(correct_target_pred_class_0)/len(labels_target[labels_target==0])
        
        #Sensitifity
        if len(labels_target[labels_target==1]) == 0:
            acc_total_target_class_1 = 0
        else:
            acc_total_target_class_1 = 100 * len(correct_target_pred_class_1)/len(labels_target[labels_target==1])

        #Balanced Accuracy Target Domain
        balanced_target_accuracy = (acc_total_target_class_0 + acc_total_target_class_1)/2


        # Separation between MMD and CE Train Phase
        if mmd_loss_flag_phase == True:
            loss = source_ce_loss + mmd_loss
        else:
            loss = source_ce_loss
        
        return loss, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, balanced_target_accuracy, class_0_source_fc2, class_1_source_fc2, class_0_target_fc2, class_1_target_fc2
