# german-traffic-sign-recognition

Dataset : https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

German Traffic Sign Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB)
. The model learns to identify and classify images of traffic signs into their respective categories.

📌 Features

Preprocessing of the GTSRB dataset (resizing, normalization, one-hot encoding).

Implementation of a CNN architecture using TensorFlow/Keras.

Training, validation, and evaluation on test data.

Visualization of training history (accuracy & loss).

Model evaluation with classification report and confusion matrix.

Prediction on custom images.

🧠 Model Architecture

The CNN is designed with:

Convolutional Layers for feature extraction

MaxPooling Layers for dimensionality reduction

Dropout Layers to reduce overfitting

Fully Connected Layers for classification

Softmax Output Layer for multi-class prediction
