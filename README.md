# Image-Classifting-Web-App
Classifies images using Flask Framework with TensorFlow

Implement Pre-process Functions
 - Normlize, One hot encoding
 - Data split in train/dev/test sets

Build the network
 - CNN model and its cost function & optimizer

Train the Neural Network
 - Hyper parameters
 - Train the model

 Test Model(prediction)
 
 
 **Idea**: 
- Classifies images using Flask Framework with TensorFlow

**How**:
- Implement pre-process functions. Through normalization and one hot encoding
- Data split in train/dev/test sets
- Train the model for neural network.


**Difficulties**
- Epochs more than 7 led to overfitting with val_acc going down.
- Increased batch_size from default=32 to increase accuracy
- Confusing about the following Neural Network's architecture:
  1. Convolutional Layer with relu activation function
  2. Maxpooling layer
  3. Convolutional Layer with relu activation functoin again
  4. Maxpooling layer
  5. Flatten layer
  6. Dense layer with relu activation function
  7. Dense layer with softmax activation function
 
