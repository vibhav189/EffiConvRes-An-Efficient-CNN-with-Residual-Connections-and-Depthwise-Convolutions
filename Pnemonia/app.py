import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'model.h5'))

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['Normal', 'Pneumonia']


def predict(filename, model):
    img = load_img(filename, target_size=(128, 128))
    img = img_to_array(img)
    img = img.reshape(1, 128, 128, 3)

    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)

    res = result[0]
    res_argsort = res.argsort()[::-1][:2]  # Get indices of two highest probabilities

    if len(res_argsort) < 2:
        class_result = [classes[res_argsort[0]]]
        prob_result = [(res[res_argsort[0]] * 100).round(2)]
    else:
        class_result = [classes[res_argsort[0]], classes[res_argsort[1]]]
        prob_result = [(res[res_argsort[0]] * 100).round(2), (res[res_argsort[1]] * 100).round(2)]

    return class_result, prob_result


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "prob1": prob_result[0],
                    "class2": class_result[1] if len(class_result) > 1 else "",
                    "prob2": prob_result[1] if len(prob_result) > 1 else "",
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accessible or inappropriate input'

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)


        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "prob1": prob_result[0],
                    "class2": class_result[1] if len(class_result) > 1 else "",
                    "prob2": prob_result[1] if len(prob_result) > 1 else "",
                }

            else:
                error = "Please upload images of jpg, jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)


