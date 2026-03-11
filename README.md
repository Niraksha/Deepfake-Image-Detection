# Deepfake-Image-Detection
Deep learning-based system for detecting AI-generated fake images using EfficientNetB0 feature extraction and a Multilayer Perceptron classifier.

This project focuses on detecting AI-generated fake images using deep learning techniques. 
The system uses EfficientNetB0 as a feature extractor and a Multilayer Perceptron (MLP) classifier to distinguish between real and manipulated images.

The goal of this project is to help identify deepfake content and reduce the spread of misinformation on digital platforms.

## Project Overview
With the rise of Generative Adversarial Networks (GANs), fake images can now be generated that are extremely realistic and difficult for humans to detect.

This project builds an automated deep learning system capable of detecting such manipulated images by learning patterns and inconsistencies present in fake images.

The system performs binary classification:

- **Real Image**
- **Fake Image**

## Objectives

- Detect AI-generated deepfake images automatically
- Improve trust in digital visual content
- Assist social media platforms in detecting fake media
- Support law enforcement and media verification systems

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Scikit-learn
- Matplotlib

## Dataset

The dataset contains images of human faces categorized into two classes:

- **Real Faces** – Authentic images
- **Fake Faces** – AI-generated or manipulated images

Dataset characteristics:

- Image size: **256 × 256 pixels**
- Organized into **train, validation, and test datasets**
- Balanced dataset for reliable training

## Methodology

The system follows the pipeline below:

1. **Data Preprocessing**
   - Image resizing to 224×224
   - Normalization
   - Data augmentation (rotation, zoom, horizontal flip)

2. **Feature Extraction**
   - EfficientNetB0 used as a pretrained backbone model
   - Extracts high-level visual features from images

3. **Model Training**
   - Multilayer Perceptron (MLP) classifier
   - Dropout and Batch Normalization to reduce overfitting
   - Optimized using RMSprop optimizer

4. **Evaluation**
   - Accuracy
   - Confusion Matrix
   - Classification Report
   - ROC-AUC Score

## Model Architecture

Feature Extractor:
EfficientNetB0 (Pretrained on ImageNet)

Classifier:
- Dense(128) + ReLU
- Dense(64) + ReLU
- Dense(32) + ReLU
- Output Layer (Softmax)

Regularization:
- Batch Normalization
- Dropout Layers

## Results

Model Performance:

- **Training Accuracy:** 75.50%
- **Testing Accuracy:** 69.33%
- **ROC-AUC Score:** 0.77

The results show that the model is capable of detecting fake images with moderate accuracy and good generalization performance.

## Model Comparison

Several models were tested before selecting the final model:

| Model | Train Accuracy | Test Accuracy |
|------|---------------|--------------|
| MLP | 75.50% | 69.33% |
| LightGBM | 78.56% | 63.00% |
| XGBoost | 89.95% | 64.35% |
| AdaBoost | 68.27% | 61.73% |

The **MLP model performed best on unseen data**, making it the final selected model.


## Applications

- Social media fake content detection
- Digital media verification
- Online misinformation prevention
- Law enforcement investigation tools

---

## Future Improvements

Possible future enhancements include:

- Explainable AI techniques (Grad-CAM, SHAP)
- Integration with social media upload filters
- Multimodal deepfake detection (image + video + audio)
- Real-time deployment on web platforms



This project is developed for academic and research purposes.
