import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

classes_cifar_10 = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

                    
classes_cifar_100 = {
    0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver',
    5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle',
    10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly',
    15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle',
    20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach',
    25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dinosaur',
    30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox',
    35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard',
    40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard',
    45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain',
    50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter',
    56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree', 60: 'plain',
    61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon',
    67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark',
    74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider',
    80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 
    85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor',
    90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale',
    96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm',
}

classes_Pneumonia = {0: 'Normal', 1: "Pneumonia"}

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3)
    return image

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def inference_cifar_10(image_path, model_path):
    #load image and model
    image = load_image(image_path)
    model = load_model(model_path)
    if image.shape[0]!=32 and image.shape[1]!=32:
        image = tf.image.resize(image, [32,32])
    image = tf.expand_dims(image, axis=0)
    preds = model.predict(image)
    preds = tf.math.argmax(preds, axis=1)
    return classes_cifar_10[preds.numpy()[0]]

def inference_cifar_100(image_path, model_path):
    #load image and model
    image = load_image(image_path)
    model = load_model(model_path)
    if image.shape[0]!=32 and image.shape[1]!=32:
        image = tf.image.resize(image, [32,32])
    image = tf.expand_dims(image, axis=0)
    preds = model.predict(image)
    preds = tf.math.argmax(preds, axis=1)
    return classes_cifar_100[preds.numpy()[0]]

def inference_Pneumonia(image_path, model_path):
    #load image and model
    image = load_image(image_path)
    model = load_model(model_path)
    if image.shape[0]!=128 and image.shape[1]!=128:
        image = tf.image.resize(image, [128,128])
    image = tf.expand_dims(image, axis=0)
    preds = model.predict(image)
    if preds>0.5:
        return classes_Pneumonia[1]
    else:
        return classes_Pneumonia[0]