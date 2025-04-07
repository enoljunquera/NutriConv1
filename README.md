# NutriConv  
**Multitask CNN for Food Classification and Weight Estimation**

This repository contains the source code and supplementary files for **NutriConv**, a multitask convolutional neural network developed for simultaneous food classification and weight estimation from images. The implementation is part of the study:

**"NutriConv: A Convolutional Approach for Digital Dietary Tracking trained on EFSA’s PANCAKE Dataset"**

---

## Repository Structure

- `ClasReg_CV.py`: Main training and evaluation script using **Cross Validation**.
- `ClasReg_FS.py`: Variant using a **Fully Shuffled** data split.
- `HyperparamMulticlass.txt`: Hyperparameter configuration file used by the scripts.
- `tensorflow25.yaml`: Conda environment file containing all required dependencies.
- `Clas+Reg - CV_alpha099/`: Folder containing:
  - The best-performing model trained with **α = 0.99**.
  - A CSV file with the model’s test set predictions.

> **Note**: The dataset used in this project (derived from EFSA's PANCAKE project) is **not included** in this repository and must be obtained separately.

---

## Model Description

**NutriConv** is a multitask convolutional neural network with two output branches: one for classification and one for regression. It is trained using a hybrid loss function (Cross-Entropy for classification and Mean Squared Error for regression). The architecture was optimized using data adapted from the PANCAKE project, a validated nutritional study from the European Food Safety Authority (EFSA).

The best trade-off was achieved with **α = 0.99**, favoring classification while maintaining strong regression performance.

---

## Requirements

It is recommended to set up the environment using Anaconda Navigator.
