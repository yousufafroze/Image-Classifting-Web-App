'''
Implement Pre-process Functions
 - Normlize, One hot encoding
 - Data split in train/dev/test sets

Build the network
 - CNN model and its cost function & optimizer

Train the Neural Network
 - Hyper parameters
 - Train the model

 Test Model(prediction)
'''


# In[1]:


# Project Description: This program classifies images


# In[2]:


# # Import libraries / packages
# from keras.datasets import cifar10
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# import numpy as np
# # Load the data

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()


# # In[3]:


# # Print the data types
# print(type(x_train))
# print(type(y_train))
# print(type(x_test))
# print(type(y_test))


# # In[9]:


# # Get the shapes
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test.shape)


# # In[17]:


# # Take a look at the first image (at index = 0) in the training datset
# x_train.shape


# # In[24]:


# # Show image as picture
# img = plt.imshow(x_train[0])


# # In[26]:


# # Print the label of the image
# print("The label is:", y_train[0])


# # In[28]:


# # One-Hot Encoding: Convert the labels into a set of 10 numbers
# #  to input into the neural network
# from keras.utils import to_categorical
# y_train_one_hot = to_categorical(y_train)
# y_test_one_hot = to_categorical(y_test)


# # In[30]:


# y_train_one_hot.shape


# # In[31]:


# # Print the new labels in the training data set
# print(y_train_one_hot)


# # In[34]:


# # Print an example of the new labels
# print("The one hot label is:", y_train_one_hot[0])


# # In[38]:


# # Normalize the pixels in the images to be values between 0 and 1

# x_train = x_train / 255
# x_test = x_test / 255


# # In[113]:


# # Build the CNN (Convolution Neural Network)

# # Create the architecture
# model = Sequential()

# # Convolution layer
# model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32, 32, 3)))

# # Maxpooling layer
# model.add(MaxPooling2D(pool_size=(2,2)))

# # Convolution layer
# # Increase filter or else performance affected significantly
# model.add(Conv2D(64, (5,5), activation='relu', input_shape=(32, 32, 3)))

# # Maxpooling layer
# model.add(MaxPooling2D(pool_size=(2,2)))

# # Flatten layer
# model.add(Flatten())

# model.add(Dense(1000, activation ='relu'))
# model.add(Dense(10, activation='softmax'))


# # In[114]:


# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )


# # In[115]:


# # Train/Fit the model
# # Epochs more than 7 leads to overfitting with val_acc going down.
# # Increased batch_size from default=32 to increase accuracy
# hist = model.fit(x_train, y_train_one_hot, batch_size = 256, epochs=10, validation_split=0.3)


# In[116]:


# # Get the models accuracy
# model.evaluate(x_test, y_test_one_hot)[1]


# # In[117]:


# hist.history


# # In[123]:


# # Visualize the Model's Accuracy
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(['Train', 'Val'], loc='upper left')


# # In[119]:


# # Visualize the Model's Loss
# plt.plot(hist.history['loss'], 'y')
# plt.plot(hist.history['val_loss'], 'r')
# plt.title("Model Loss")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(['Train', "Val"], loc='upper left')


# In[297]:


# from IPython.display import Image
# img = Image("Desktop/cat.jpeg")


# # In[298]:


# img


# In[299]:


# Convert the image to an array-like image
# import cv2
# img = cv2.imread("Desktop/cat.jpeg", flags=cv2.IMREAD_COLOR)


# # In[300]:


# # Resize the image
# from skimage.transform import resize
# my_image_resized = resize(img, (32,32,3))
# img = plt.imshow(my_image_resized)


# # In[301]:


# # Get the probabilities for each class
# import numpy as np
# probabilities = model.predict(np.array(my_image_resized.reshape(1,32,32,3)))[0]


# # In[302]:




# # In[306]:


# print("Most likely class:", number_to_class[index[9]], '--probability:', probabilities[index[9]])

# print("Second most likely class:", number_to_class[index[8]], '--probability:', probabilities[index[8]])

# print("Third most likely class:", number_to_class[index[7]], '--probability:', probabilities[index[7]])

# print("Fourth most likely class:", number_to_class[index[6]], '--probability:', probabilities[index[6]])

# print("Fifth most likely class:", number_to_class[index[5]], '--probability:', probabilities[index[5]])


# # In[304]:


# # # Save the model
# model.save('my_model.h5')


# # In[305]:


# # Load the model
# from keras.models import load_model
# model = load_model('my_model.h5')


# # In[ ]:




