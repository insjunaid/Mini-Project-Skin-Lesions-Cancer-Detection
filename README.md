# Skin Lesions Detection Using Deep Learning

![Screenshot 2024-12-19 004521](https://github.com/user-attachments/assets/da8226d4-2c15-409c-9b75-7040b7bb82b6)




This repository contains a PyTorch-based implementation for detecting various types of skin lesions using a fine-tuned EfficientNet-B4 model. The project includes a Flask web application to upload skin lesion images and obtain predictions with recommendations for further action.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
  - [Data Augmentation](#data-augmentation)
  - [Training Script](#training-script)
- [Evaluation](#evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [ROC Curve](#roc-curve)
- [Web Application](#web-application)
  - [Flask Application](#flask-application)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#Conclusion)

---

## Introduction

Skin cancer is one of the most common cancers worldwide, and early detection is critical for effective treatment. This project utilizes deep learning to classify skin lesion images into seven categories, including both benign and malignant types, providing actionable insights.

## Dataset

The dataset, sourced from [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), includes images of skin lesions categorized into the following classes:

1. **Actinic keratoses and intraepithelial carcinoma (akiec)** - Malignant
2. **Basal cell carcinoma (bcc)** - Malignant
3. **Benign keratosis-like lesions (bkl)** - Benign
4. **Dermatofibroma (df)** - Benign
5. **Melanoma (mel)** - Malignant
6. **Melanocytic nevi (nv)** - Benign
7. **Vascular lesions (vasc)** - Benign

The dataset is split into training (70%), validation (15%), and testing (15%) sets. Data balancing techniques were used to ensure equal representation of all classes.

## Requirements

Dependencies:

- Python 3.8+
- PyTorch
- Torchvision
- Flask
- Pillow
- scikit-learn
- Matplotlib
- Seaborn
- TQDM

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Model Architecture

The project uses a fine-tuned EfficientNet-B4 model with the following modifications to the classifier layer:

- `Linear(in_features, 512)`
- `ReLU()`
- `Dropout(0.5)`
- `Linear(512, 7)`

The model is pretrained on ImageNet and fine-tuned on the skin lesion dataset.

## Training

### Data Augmentation

The training data is augmented with transformations, including:

- Random horizontal and vertical flips
- Random rotation (up to 30 degrees)
- Random resized crop (scale: 70-100%)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization

### Training Script

The training script implements:

- Early stopping with a patience of 6 epochs
- Weighted loss function to address class imbalance
- Learning rate scheduler to reduce learning rate on plateau

Run the training script:

```bash
python train.py
```

## Evaluation

### Confusion Matrix

The confusion matrix is plotted to visualize the model’s performance on each class.

### ROC Curve

ROC curves are generated for all classes to evaluate the classifier’s ability to distinguish between categories.

## Web Application

### Flask Application

The Flask app allows users to upload an image of a skin lesion and provides the following:

- Predicted class
- Malignancy status (benign or malignant)
- Recommendation based on the result

Run the Flask application:

```bash
python app.py
```

Access the app at `http://127.0.0.1:5000/`.

## Results

- **Accuracy**: The model achieves a high accuracy on the test set.
- **AUC-ROC**: High AUC scores for malignant classes indicate strong performance.

## Installation

 
  Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the dataset in the appropriate folder structure:

   ```
   dataset/
       train/
           class_1/
           class_2/
           ...
       val/
           class_1/
           class_2/
           ...
       test/
           class_1/
           class_2/
           ...
   ```

2. Run the training script to train the model.
3. Launch the Flask app to predict skin lesions.

## Conclusion
This project demonstrates the potential of deep learning in medical image analysis, highlighting its role in aiding early detection and diagnosis.

Images of Types of Skin Lesions

![l](https://github.com/user-attachments/assets/ad1e339b-16d4-4677-96a0-47db7dd77a2e)

## Acknowledgement
Under guidance of  [Dr Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu)


