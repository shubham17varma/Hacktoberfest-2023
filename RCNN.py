import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

input_image = layers.Input(shape=(224, 224, 3), name='input_image')
roi_input = layers.Input(shape=(4,), name='input_roi')

roi_layer = layers.Lambda(lambda x: tf.image.crop_and_resize(x[0], x[1], [0]*tf.shape(x[1])[0], (7, 7)), name='roi_layer')([input_image, roi_input])

roi_features = base_model(roi_layer)

flatten_layer = layers.Flatten(name='flatten')(roi_features)
fc1 = layers.Dense(256, activation='relu', name='fc1')(flatten_layer)
classification_head = layers.Dense(num_classes, activation='softmax', name='classification_head')(fc1)
regression_head = layers.Dense(4, activation='linear', name='regression_head')(fc1)

rcnn_model = keras.Model(inputs=[input_image, roi_input], outputs=[classification_head, regression_head])

rcnn_model.compile(optimizer='adam',
                   loss={'classification_head': 'categorical_crossentropy', 'regression_head': 'mse'},
                   loss_weights={'classification_head': 1.0, 'regression_head': 1.0},
                   metrics={'classification_head': 'accuracy', 'regression_head': 'mae'})

image_path = 'image.jpg'
roi_coordinates = np.array([[x1, y1, x2, y2]])
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)

classification_preds, regression_preds = rcnn_model.predict([np.expand_dims(image, axis=0), roi_coordinates])
