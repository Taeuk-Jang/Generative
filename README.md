# Generative
Generative models for fair disentanglement.

## Information about the source code:
util.py : tnse visualization and save utils.
module.py : contain models for each data domains.
train_baseline.py : Training script for baseline VAE model.

train_fair-3.py : This is on-progress script for HQ images.

There are three experiments demonstrated in each notebook files.

## To download datasets:
* Tabular dataset : https://github.com/Trusted-AI/AIF360
Extract aif360 folder and paste into this repo.


## To download Pre-trained models:
* MNIST-USPS : https://purdue0-my.sharepoint.com/:u:/g/personal/jang141_purdue_edu/ERV3B8F_WhBLpvctigizEgIB_uLNTM1EO2z91Wez_MaG3g?e=d4JIGE
* CelebA : https://purdue0-my.sharepoint.com/:u:/g/personal/jang141_purdue_edu/EffRnHfarkdAktBc1acUJMYB0K86Ti2-Fa8cpO8ujjLLRA?e=IqoOVR

## Prerequisites
* Python 3.6
* PyTorch


The structure of VAE for CelebA is adopted from previous work (https://github.com/hhb072/IntroVAE)

