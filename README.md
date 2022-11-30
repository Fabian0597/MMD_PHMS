# Master's Thesis: Intelligent Ball Screw Fault Diagnosis Using Deep Learning Based Domain Adaptation and Transfer Learning

## Introduction
This Repository contains the Code for a deep learning-based domain adaptation Health Monitoring System for industrial applications. Since ball screw feed drives are widely used components in industrial machines, the developed system was evaluated on their degradation monitoring. In the course of the thesis data was recorded at the ball screw feed drives mounted in the machine tool's moving hanger assembly of a DMG DMC 55H duoblock milling machine of the manufacturer DMG Mori. Due to varying working conditions, prognostics and health management systems must be developed robust enough to handle continuous changes in the fault characteristics of industrial machines. Throughout the proposed model training, the domain shift in the latent feature spaces of the model is measured and reduced by a maximum mean discrepancy loss. 

## Model & Model Training
The architecture of the proposed PHM model, which is visualized in fig.1, consists of an one-dimensional CNN and a subsequent classifier. The CNN extracts expressive features, which are later used by the classifier to predict the health condition of the ball screw drives. After iteratively applying convolutional, pooling and batch normalization layers, the output of the CNN is flattened and normalized to an one-dimensional vector. This vector is fed to the subsequent classifier. The repetitive model training is separated into two phases. In the first phase, a weighted average of source cross entropy loss and maximum mean discrepancy loss is used to optimize the whole network. In the second phase, only the cross entropy loss is applied to optimize the final layers of the classifier. 

<table style="margin-left: auto; margin-right: auto; table-layout: fixed; width: 800px">
  <tr>
    <td style="margin-left: 100px; width: 800px"> <img src="ressources/proposed_model.png" width='500'></td>
  </tr>
  <tr>
    <td style="width: 800px"" valign="top"> <b>Fig.1:</b> Model Architecture and Traing Phases .
  </tr>
</table>

The maximum mean discrepancy loss estimates the domain discrepancy in the latent feature maps of the neural network. The maximum mean discrepancy loss facilitates the extraction of domain-invariant features. The domain discrepancy is measured as the squared distance between the distribution kernel embeddings in the reproducing kernel Hilbert space. The maximum mean discrepancy loss is applied in several layers of the CNN and classifier. The cross entropy loss, which is applied in the final layer of the classifier, optimizes the model to increase the classification accuracy on the source domain data. The application of the different losses during the training is visualized in fig.2.
                                          
<table style="margin-left: auto; margin-right: auto; table-layout: fixed; width: 800px">
  <tr>
    <td style="margin-left: 100px; width: 800px"> <img src="ressources/MMD_loss_visualization.png" width='650'></td>
  </tr>
  <tr>
    <td style="width: 800px"" valign="top"> <b>Fig.1:</b> Model Architecture and Traing Phases .
  </tr>
</table>

## Research Questions


### Influence of the GAMMA Choice on the Domain Adaptation Performance

Since the source and target domains are correlated to some extent, the network itself can extract domain-independent features. The powerful CNN learned from the source domain can also increase the model performance on the target domain. At the same time, features that are too sensitive to the source domain can reduce the model performance on the target domain. To counteract that phenomenon, domain adaptation approaches can help to transfer knowledge learned from the source to the target domain. However, one has to pay attention to not transfer noise or irrelevant information since this destroys the structure of the source and target domain data and makes the classification task even more difficult. For this reason, it is essential to precisely balance the maximum mean discrepany and cross entropy loss. This thesis investigates the effects of different weighting factors, called GAMMA, on the model training.

#### Results
<table style="margin-left: auto; margin-right: auto; table-layout: fixed; width: 100%">
  <tr>
    <td style="width: 48%;"> <img src="resources/GAMMA_Influence_dummy_distribution/Dummy_distribution_0_GAMMA_0_001.png" width='300'></td>
    <td style="width: 48%;"> <img src="resources/GAMMA_Influence_dummy_distribution/Dummy_distribution_0_GAMMA_0_1.png" width='300'></td>
    <td style="width: 48%;"> <img src="resources/GAMMA_Influence_dummy_distribution/Dummy_distribution_0_GAMMA_20.png" width='300'></td>
  </tr>
  <tr>
    <td style="width: 48%;" valign="top"> <b>Fig.2:</b> 'Exponential' soft padding (0.2 m).
    </td>
    <td style="width: 48%;" valign="top">  <b>Fig.3:</b> 'Exponential' soft padding (1.0 m).
    </td>
    <td style="width: 48%;" valign="top">  <b>Fig.4:</b> 'Linear' soft padding (1.0 m).
    </td>
  </tr>
</table>

### Domain Adaptation Performance of the Labeled Maximum Mean Discrepancy Loss

In this thesis a novel labeled maximum mean discrepancy loss was developed, which minimizes the domain discrepancy between samples of the same class and maximizes the domain discrepancy between samples of different classes. This novel maximum mean discrepancy loss directly considers the labels of the source and target domain. The target labels are not used in the cross entropy loss. The advantages and disadvantages of the labeled maximum mean discrepancy loss over the unlabeled maximum mean discrepancy loss are further analyzed in this thesis.

### Influence of the Latent Feature Space Choice on the Domain Adaptation Performance
Most domain adaptation approaches reduce the domain discrepancy in the task-specific layers and use a shared CNN backbone across all domains. Throughout the neural network, the feature maps extract information with different levels of abstraction. Since feature maps influence all subsequent ones, propagating the biased data through the neural network facilitates the domain shift. Reducing the domain discrepancy in only task-specific layers might minimize but not completely eliminate it. This thesis investigates how applying the maximum mean discrepancy loss in different model layers can improve the domain discrepancy reduction. In this context, a particular focus lies on the layers of the CNN.

