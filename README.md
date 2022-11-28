# Master Thesis: Intelligent Ball Screw Fault Diagnosis Using Deep Learning Based Domain Adaptation and Transfer Learning

## Introduction
This Repository contains the Code for a deep learning-based domain adaptation Health Monitoring System for industrial applications. Since ball screw feed drives are widely used components in industrial machines, the developed system was evaluated on their degradation monitoring. For this reason, data was recorded at the ball screw feed drives mounted in the machine tool's moving hanger assembly of a DMG DMC 55H duoblock milling machine of the manufacturer DMG Mori. Due to varying working conditions, prognostics and health management systems must be developed robust enough to handle continuous changes in the fault characteristics of industrial machines. Throughout the proposed model training, the domain shift in the latent feature spaces of the model is measured and reduced by the maximum mean discrepancy.

## Model
The architecture of the proposed PHM model consists of an one-dimensional CNN and a subsequent classifier. The CNN extracts expressive features, which are later used by the classifier to predict the health condition of the ball screw drives. After iteratively applying convolutional, pooling and batch normalization layers, the output of the CNN is flattened and normalized to an one-dimensional vector. This vector is fed to the subsequent classifier. The repetitive model training is separated into two phases. In the first phase, a weighted average of source cross entropy loss and maximum mean discrepancy loss is used to optimize the whole network. In the second phase, only the cross entropy loss is applied to optimize the final layers of the classifier. 

<table style="margin-left: auto; margin-right: auto; table-layout: fixed; width: 800px">
  <tr>
    <td style="margin-left: 100px; width: 800px"> <img src="resources/proposed_model.pdf" width='300'></td>
  </tr>
  <tr>
    <td style="width: 800px"" valign="top"> <b>Fig.5:</b> Local costmap (green) detects a new obstacle.
  </tr>
</table>

![title](resources/proposed_model.pdf)
