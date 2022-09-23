import tensorflow as tf
from keras.applications import ResNet50V2

def build_model():
    base_model = ResNet50V2(include_top=False, input_shape=(480, 480, 3))

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

model = build_model()
model.load_weights("mobile/weight/resnet.h5")
model.save('mobile/model/ResNet_Model.h5')

tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tf_lite_converter.convert()
open("mobile/weight/ResNet.tflite", "wb").write(tflite_model)
    