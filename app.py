from __future__ import division, print_function

from flask import Flask, request, render_template

import os
import numpy as np
from werkzeug.utils import secure_filename

from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = "C:\\Users\\ayushi jajoo\\Desktop\\vgg19\\static"
model = load_model("vgg19.h5")


def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# C:\Users\ayushi jajoo\Desktop\vgg19

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            predict_class = predict(file_path, model)
            pred_class = decode_predictions(predict_class, top=1)
            print(file_path)
            print(filename)
            return render_template("index.html", pre=pred_class[0][0][1], image_loc=filename)
    return render_template("index.html", pre=0)


if __name__ == "__main__":
    app.run()
