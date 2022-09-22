import io
import keras
from PIL import Image, ImageOps
from keras.applications import VGG16, ResNet50V2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
import io

def vgg_model():
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(480, 480, 3))

    for layer in base_model.layers[:(len(base_model.layers) // 3 * 2)]:
        layer.trainable = False
        
    model = tf.keras.Sequential([
        base_model, 
        tf.keras.layers.GlobalAveragePooling2D(), 
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def resnet_model():
    base_model = ResNet50V2(include_top=False, weights="imagenet", input_shape=(480, 480, 3))

    for layer in base_model.layers[:(len(base_model.layers) // 3 * 2)]:
        layer.trainable = False
        
    model = tf.keras.Sequential([
        base_model, 
        tf.keras.layers.GlobalAveragePooling2D(), 
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def prediction(img):
    # Load the model
    model = resnet_model()
    model.load_weights('web/weight/resnet.h5')
    
    image = load_img(img, target_size=(480, 480))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    
    # # run the inference
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_gen = test_datagen.flow(image, batch_size=1)
                        
    prediction = model.predict(test_gen)
    
    # return np.argmax(prediction) # return position of the highest probability
    return prediction[0]
