from python_application import app 
from flask import render_template, flash, redirect
from keras.models import load_model
import python_application.classifier
import numpy as np

import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



@app.route('/', methods=['GET'])
def index():
	# Main page
	return render_template('index.html')




@app.route('/classify', methods = ['POST', 'GET'])
def classify():
	if request.method == 'POST':
		# Get the file from post request
				# Get the file from post request
		f = request.files['image']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		my_image_resized = image.load_img(file_path, target_size=(32, 32, 3))

		# Get the probabilities for each class
		model = load_model('my_model.h5')

		probabilities = model.predict(np.array(my_image_resized.reshape(1,32,32,3)))


		number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		index = np.argsort(probabilities)


		result = "Most likely class:" +  number_to_class[index[9]] +  '--probability:' +  probabilities[index[9]]

		# # Process your result for human
		# # pred_class = preds.argmax(axis=-1)            # Simple argmax
		# pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
		# result = str(pred_class[0][0][1])               # Convert to string
		return result
	return None













# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds



# if __name__ == '__main__':
#     # app.run(port=5002, debug=True)

#     # Serve the app with gevent
#     http_server = WSGIServer(('0.0.0.0', 5000), app)
#     http_server.serve_forever()
