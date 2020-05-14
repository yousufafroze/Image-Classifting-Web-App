# Image-Classifting-Web-App
 
 **Idea**: 
- Classifies images using Flask Framework with TensorFlow

**How**:
- Classifies images with 68% accuracy from the CIFAR-10 dataset (some animals, objects etc) using a CNN built in TensorFlow
- Deployed the model through a web application using Flask for Python

**Summary**:
- Implement pre-process functions. Through normalization and one hot encoding
- Data split in train/dev/test sets
- Train the model for neural network.

**Difficulties**
- Epochs more than 7 led to overfitting with val_acc going down.
- Increased batch_size from default=32 to increase accuracy
- Making the following Neural Network's architecture to reach 68% accuracy:
  1. Convolutional Layer with relu activation function
  2. Maxpooling layer
  3. Convolutional Layer with relu activation functoin again
  4. Maxpooling layer
  5. Flatten layer
  6. Dense layer with relu activation function
  7. Dense layer with softmax activation function
 
