from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

class_name = ['Dog','Cat']
with sess.as_default():
    with graph.as_default():
        models = load_model('DogCat.h5')
###############################
app = Flask(__name__)
# CORS(app)
# app.config['CORS_HEADERS'] = 'Content_Type'

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


generator = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)

@app.route('/', methods=['POST'])
# @cross_origin(origins='*')
def predict():
    global sess, graph, models
    f = request.files['file']
    img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img =cv2.resize(img,(300,300))
    image = img_to_array(img)
    image = np.expand_dims(image, axis=0)
    img_array = generator.standardize(image)
    # img_array = img_array / 255.0
    # print(img_array)
    with sess.as_default():
        with graph.as_default():
            predict = models.predict(img_array)
    print(predict)
    # return render_template('index.html', prediction=class_name[np.argmax(predict, axis=1)])
    return render_template('index.html', prediction=4)

if __name__ == '__main__':
    app.run(debug=True, port=3000)