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

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jif'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
    'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
    'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
    'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm'
]


def predict(filename, model):
    img = Image.open(filename)
    img = img.resize((32, 32))
    img = img.convert('RGB')
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)

    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(100):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]

    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i] * 100).round(2))
        class_result.append(dict_result[prob[i]])

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
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
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
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
