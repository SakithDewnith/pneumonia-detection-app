# Chest X-Ray Pneumonia Detection using Deep Learning

An end-to-end data science project that applies computer vision and deep learning techniques to classify chest X-ray images as **Normal** or **Pneumonia**.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

</p>

🚀 Live Demo:
https://sakith-pneumonia-detection.streamlit.app/


### Dashboard Images
<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/59a0bc4f-c844-49a1-b210-40171edef990" width="550" alt="Screenshot 1"><br>
        <sub><b>State: idle</b></sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/34161686-00b3-456b-839c-64a09ad0dc00" width="550" alt="Screenshot 2"><br>
        <sub><b>State: Pneumonia Positive</b></sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/d77d4a51-eac7-40a9-b48d-1fa56965c5f0" width="550" alt="Screenshot 3"><br>
        <sub><b>State: Pneumonia Negative</b></sub>
      </td>
    </tr>
  </table>
<br>

## Executive Summary

<p>
Pneumonia is a serious respiratory disease where early detection plays an important role in effective treatment.This project develops an automated chest X-ray classification system using a Convolutional Neural Network (CNN) to assist pneumonia screening.

The workflow includes:
</p>

<ul style="margin-left: 25%; text-align: left;">
    <li> Preparing and balancing medical image datasets for reliable model training.</li>
    <li> Improving X-ray image quality using enhancement and normalization techniques.</li>
    <li> Training a deep learning classification model to detect pneumonia patterns.</li>
    <li> Measuring performance using healthcare-focused classification metrics.</li>
    <li> Deploying the trained model through an interactive Streamlit application.</li>
</ul>

<p>
The final model achieved:
</p>

<ul style="margin-left: 25%; text-align: left;">
    <li> Achieved 85.1% accuracy on unseen chest X-ray images.</li>
    <li> Achieved 90% recall to minimize missed pneumonia cases.</li>
    <li> Achieved 93.4% AUC-ROC, demonstrating strong classification performance.</li>
</ul>

High recall was prioritized because reducing missed pneumonia cases is more important than minimizing false alarms.
<br>
<br>

## Healthcare Problem

Traditional pneumonia diagnosis relies on radiologists manually analyzing chest X-rays. However, healthcare systems may face challenges such as:

<ul style="margin-left: 25%; text-align: left;">
    <li>Shortage of radiologists can delay diagnosis, especially in high-demand healthcare environments.</li>
    <li>Manual analysis of chest X-rays requires significant time and expert attention.</li>
    <li>Differences in clinical experience may lead to variations in image interpretation.</li>
    <li>Growing numbers of medical images require efficient AI-assisted screening solutions.</li>
</ul>

The objective of this project is to develop a machine learning solution that can assist healthcare professionals by providing fast and consistent pneumonia predictions from chest X-ray images.
<br>
<br>

## Methodology

The project follows an end-to-end data science workflow:

### 1. Data Preparation

The dataset was created by combining:
<ul style="margin-left: 25%; text-align: left;">
    <li>NIH ChestX-ray14</li>
    <li>Kaggle Chest X-Ray Images (Pneumonia)</li>
</ul>

Dataset distribution:
| Dataset Split | Normal | Pneumonia |
| :--- | :---: | :---: |
| **Training** | 4,563 | 4,563 |
| **Validation** | 566 | 543 |
| **Testing** | 571 | 570 |

A balanced dataset was used to reduce class imbalance and improve model generalization.

### 2. Data Preprocessing

The images were processed using:
<ul style="margin-left: 25%; text-align: left;">
    <li>Grayscale conversion</li>
    <li>Image resizing (224×224))</li>
    <li>CLAHE contrast enhancement</li>
  <li>Pixel normalization</li>
  <li>Data augmentation</li>
</ul>

These steps improved image quality and helped the model learn meaningful patterns.

### 3. Model Development

A CNN classification model was developed using TensorFlow/Keras.

The model includes:
<ul style="margin-left: 25%; text-align: left;">
    <li>Convolutional layers for feature extraction</li>
    <li>Batch normalization for stable learning</li>
    <li>Dropout for reducing overfittingt</li>
  <li>Dense layers for classification</li>
  <li>Sigmoid output for binary prediction</li>
</ul>

#### Skills & Technologies

| Category | Technologies & Concepts |
| :--- | :--- |
| **Programming** | Python |
| **Data Processing** | NumPy, Pandas, OpenCV |
| **ML / Deep Learning** | TensorFlow, Keras, CNN, Binary Classification, Model Evaluation |
| **Visualization** | Matplotlib |
| **Deployment** | Streamlit |


### 4. Model Evaluation

The model was evaluated using:


| Metric | Score |
| :--- | :---: |
| **Accuracy** | 85.1% |
| **Precision** | 82.0% |
| **Recall** | 90.0% |
| **F1 Score** | 86.0% |
| **Specificity** | 80.6% |
| **AUC-ROC** | 93.4% |
<br>

## Results and Insights
The analysis showed that the CNN model was able to identify pneumonia patterns effectively from unseen chest X-ray images.

Key findings:

<ul style="margin-left: 25%; text-align: left;">
    <li>Achieved 90% recall, reducing the risk of missed pneumonia cases.</li>
    <li>Achieved 93.4% AUC-ROC, demonstrating effective separation between Normal and Pneumonia classes.</li>
    <li>CLAHE preprocessing improved contrast and helped highlight important X-ray features.</li>
    <li>Data augmentation helped reduce overfitting and improved model robustness on unseen images.</li>
    <li>The trained CNN model was deployed as an interactive Streamlit application for real-time predictions.</li>
</ul>
<br>

## Deployment

A Streamlit application was developed to make the model accessible.

Users can:

<ol style="margin-left: 25%; text-align: left;">
    <li>Upload a chest X-ray image through the Streamlit interface.</li>
    <li>Apply image preprocessing steps automatically before prediction.</li>
    <li>Receive a classification result as Normal or Pneumonia.</li>
    <li>View the model's prediction confidence level.</li>
</ol>
<br>

## Next Steps

Future improvements:

<ul style="margin-left: 25%; text-align: left;">
    <li>Apply advanced pretrained models such as ResNet50, EfficientNet, and DenseNet.</li>
    <li>Add Grad-CAM visualizations to improve model interpretability.</li>
    <li>Validate model performance using larger and more diverse datasets.</li>
    <li>Extend the system to detect multiple lung diseases beyond pneumonia.</li>
</ul>
