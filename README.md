# Faster Cheque Clearing with AI

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Our Plan](#our-plan)
   - [CNN Approach](#cnn-approach)
   - [Siamese Network Approach](#siamese-network-approach)
   - [Text Recognition and MICR](#text-recognition-and-micr)
3. [Implementation](#implementation)
4. [Future Work](#future-work)

## Problem Statement

The aim of this project is to automate the cheque clearing process in banks. The current process involves several manual steps including technical verifications and signature verification, which require significant human capital and time. Our goal is to use rule-based and AI/ML/ICR/OCR capabilities to automate these steps, thereby reducing human efforts, processing time, and potential frauds.

## Our Plan

Our plan involves the following steps:

### CNN Approach

1. **Data Collection**: Gather a dataset of original and forged signature images.
2. **Model Selection**: Choose a suitable model for image classification tasks. You could use a Convolutional Neural Network (CNN) based model.
3. **Training**: Train your selected model on your prepared dataset. During training, the model will learn to classify signature images as either original or forged.
4. **Evaluation**: After training, evaluate your model's performance on a separate test set.

The CNN model is specifically trained on large number of dense layers to overfit the training data and match the signature exactly.

### Siamese Network Approach

1. **Data Collection**: Gather a dataset of cheques with signatures.
2. **Data Preparation**: Make a combination of original and forged signatures such that the combination of a forged and original signature is labeled as 0, and the combination of an original and original signature is labeled as 1.
3. **Model Selection**: Choose a suitable model for signature verification tasks. You could use a Siamese Network based model.
4. **Training**: Use the prepared data to batch process and train your selected model. During training, the model will learn to verify signatures.
5. **Evaluation**: After training, evaluate your model's performance on a separate test set.

### Text Recognition and MICR

1. **Data Collection**: Take images from the dataset.
2. **Preprocessing**: Resize the images to a standard size and convert them to grayscale.
3. **Background Removal**: Perform adaptive thresholding to remove the background, suspending the noises to get the foreground properly with a specified intensity.
4. **Inversion**: Use bitwise NOT operation to invert the image, resulting in a white background and black words.
5. **Bounding Box Detection**: Apply bounding box detection to the identifiable regions to extract the text.

## Implementation

We have implemented parts of this plan in separate Jupyter notebooks:

1. **CNN Image Classifier (`cnn-image-classifier.ipynb`)**: This notebook contains a Convolutional Neural Network (CNN) model for classifying images of cheques. We used the Keras library to implement a sequential model, trained it on a dataset of original and forged signature images, and evaluated its performance.

2. **Signature Verification (`signature-verification.ipynb`)**: This notebook presents a method for verifying signatures on cheques using a Siamese Network approach.

3. **Text Recognition and MICR (`text-recognition-and-micr.ipynb`)**: This notebook presents a method for recognizing and extracting text from cheques using image processing techniques and bounding box detection.

## Future Work

While we have made significant progress, our work is not yet available as a whole. We are actively working on integrating these separate parts into a single, cohesive system for faster cheque clearing. Stay tuned for updates!
