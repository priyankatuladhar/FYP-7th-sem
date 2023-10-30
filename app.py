from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model('face-shape-recognizer.h5')

FACE_SHAPES = {
    0: "Heart",
    1: "Oblong",
    2: "Oval",
    3: "Round",
    4: "Square",
}


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


# hyperlink to example html_file
@app.route('/example')
def example():
    return render_template('example.html')


# hyperlink to example html_file
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/aboutus')
def aboutus():
    return render_template('about.html')


@app.route('/products')
def products():
    return render_template('products.html')

@app.route('/productpageonee')
def productpageonee():
    return render_template('productpageone.html')

@app.route('/productpagethree')
def productpagethree():
    return render_template('productpagethree.html')

@app.route('/productspage')
def productspage():
    return render_template('productpage.html')


@app.route('/blog')
def blog():
    return render_template('blogs.html')


@app.route('/log')
def log():
    return render_template('login.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/cart')
def cart():
    return render_template('cart.html')


@app.route('/register')
def register():
    return render_template('register.html')


# allowing user to upload
@app.route('/', methods=['POST'])
def predict():
    # load and preprocess for model
    file = request.files['imagefile']
    filename = file.filename
    file_path = os.path.join('static', 'uploads', filename)

    file.save(file_path)
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (190, 250))
    img_array = img_to_array(img_resized)
    img_normalized = preprocess_input(img_array.reshape((1,) + img_array.shape))

    # Make prediction and get face shape

    pred = model.predict(img_normalized)

    face_shape = np.argmax(pred)

    # Return result
    classification = f"{FACE_SHAPES[face_shape]} ({pred[0][face_shape] * 100:.2f}%)"
    return render_template('index.html', prediction=classification, image_path=file_path)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
