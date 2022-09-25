#!/bin/bash

#for looping through different models with different configurations

#features_of_interest=("C:s_ist/X" "C:s_soll/X" "C:s_diff/X" "C:v_(n_ist)/X" "C:v_(n_soll)/X" "C:P_mech./X" "C:Pos._Diff./X"
#        "C:I_ist/X" "C:I_soll/X" "C:x_bottom" "C:y_bottom" "C:z_bottom" "C:x_nut" "C:y_nut" "C:z_nut"
#        "C:x_top" "C:y_top" "C:z_top" "D:s_ist/X" "D:s_soll/X" "D:s_diff/X" "D:v_(n_ist)/X" "D:v_(n_soll)/X"
#        "D:P_mech./X" "D:Pos._Diff./X" "D:I_ist/X" "D:I_soll/X" "D:x_bottom" "D:y_bottom" "D:z_bottom"
#        "D:x_nut" "D:y_nut" "D:z_nut" "D:x_top" "D:y_top" "D:z_top" "S:x_bottom" "S:y_bottom" "S:z_bottom"
#        "S:x_nut" "S:y_nut" "S:z_nut" "S:x_top" "S:y_top" "S:z_top" "S:Nominal_rotational_speed[rad/s]"
#        "S:Actual_rotational_speed[µm/s]" "S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]"
#        "S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]")


#Name of experiment for tensorboard
experiment_name="single_testing"

#Number of epochs
num_epochs=100

#GAMMA Specification
GAMMA=0.05 #GAMMA hyperparameter
GAMMA_reduction=1 #GAMMA reduction of each epoch GAMMA_next_epoch = GAMMA * GAMMA_reduction

#Pooling layer definition if 1 --> Pooling layer just after Conv1, if 2 --> Pooling layer just after Conv1 & Conv2 , if 3 --> Pooling layer just after Conv1 & Conv2 & Conv3
num_pool=2

#Defines which layers should be included in MMD Loss (Conv1, Conv2, Conv3, Flattend, FC1, FC2)
MMD_layer_activation_flag=( True True False True True True )

#Features which should be considered by models
feature_of_interest=( "D:P_mech./X" "S:x_top" )

#Define which BSD states should be included in source and target domain    
list_of_source_BSD_states=("1" "2" "3" "4" "10" "11" "12" "13" "19" "20" "21" "22")
list_of_target_BSD_states=("5" "6" "7" "9" "14" "15" "16" "18" "23" "24" "25" "27")

#Define binary classification task
class_0_labels=("1" "P") #classes considered in unhealthy class
class_1_labels=("2" "3") #classes considered in healthy class

#rm -r runs

python3 main.py $experiment_name $num_epochs $GAMMA $GAMMA_reduction $num_pool ${MMD_layer_activation_flag[@]} ${#feature_of_interest[@]} ${feature_of_interest[@]} ${#list_of_source_BSD_states[@]} ${list_of_source_BSD_states[@]} ${#list_of_target_BSD_states[@]} ${list_of_target_BSD_states[@]} ${#class_0_labels[@]} ${class_0_labels[@]} ${#class_1_labels[@]} ${class_1_labels[@]}
