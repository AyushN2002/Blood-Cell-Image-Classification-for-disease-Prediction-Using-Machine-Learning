# Blood-Cell-Image-Classification-for-disease-Prediction-Using-Machine-Learning
This project uses InceptionV3 and transfer learning to classify white blood cell images (Eosinophils, Lymphocytes, Monocytes, Neutrophils,Basophil). It features data augmentation, fine-tuning, and a web app for real-time prediction. Scalable and clinically relevant, it supports early disease detection and future telemedicine integration.
![image](https://github.com/user-attachments/assets/0ca7437d-7611-4743-8539-24d2d32f3e18)

# Fine Tuning
Fine tuning consists of unfreezing the entire model you obtained above (or part of it), and re-training it on the new data with a very low learning rate. This can potentially achieve meaningful improvements, by incrementally adapting the pretrained features to the new data.

In this project, I'll be using the InceptionV3 model which will use the weights from the ImageNet dataset.

Note

Setting include_top to False moves all the layer's weights from trainable to non-trainable. This is called "freezing" the layer: the state of a frozen layer won't be updated during training

GlobalAveragePooling2D -> This layer acts similar to the Max Pooling layer in CNNs, the only difference being is that it uses the Average values instead of the Max value while pooling. This really helps in decreasing the computational load on the machine while training.

Dropout -> This layer omits some of the neurons at each step from the layer making the neurons more independent from the neibouring neurons. It helps in avoiding overfitting. Neurons to be ommitted are selected at random. The rate parameter is the liklihood of a neuron activation being set to 0, thus dropping out the neuron.

Dense -> This is the output layer which classifies the image into one of the 5 possible classes. It uses the softmax function which is a generalization of the sigmoid function.

# Fitting Model

<img width="1182" alt="Screenshot 2025-07-08 at 12 16 38 PM" src="https://github.com/user-attachments/assets/87bed359-bfb6-45ae-9a63-8c5adb11d382" />

# Learning curves

![image](https://github.com/user-attachments/assets/303b21f1-b253-423c-8b24-1b3b764f1120)

# Evaluation Metrics

<img width="237" alt="Screenshot 2025-07-08 at 12 18 00 PM" src="https://github.com/user-attachments/assets/e2a21ce2-5d66-48a3-aabf-90728cdbace5" />

![image](https://github.com/user-attachments/assets/6f4ccb60-59b7-4715-a556-ec99c5a21617)

# Testing Loss & Accuracy

<img width="304" alt="Screenshot 2025-07-08 at 12 19 10 PM" src="https://github.com/user-attachments/assets/e31a3cb9-1546-4fe9-9398-e541e66b3247" />

# Outputs

<img width="602" alt="Screenshot 2025-07-08 at 1 02 45 PM" src="https://github.com/user-attachments/assets/418be4a7-2458-4ba7-bac3-5da893ea8e8c" />

<img width="661" alt="Screenshot 2025-07-08 at 1 03 38 PM" src="https://github.com/user-attachments/assets/0c639f96-a8a6-4859-80a3-dbaa7869857b" />

<img width="666" alt="Screenshot 2025-07-08 at 1 03 22 PM" src="https://github.com/user-attachments/assets/ae826f81-926c-4f47-a0cc-370c42893332" />

