import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


model = tf.keras.models.load_model('Digit_model')
model.load_weights('model_weights.h5')
model.summary()

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img).reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0

    return img

image_path = input('Enter the path to an image file: ')
img = preprocess_image(image_path + '.PNG')
prediction = model.predict(img)
digit = np.argmax(prediction)
img = np.array(img).reshape(28, 28)   
plt.imshow(img, cmap='gray')
plt.title(f'Input image and predicted digit: {digit}')
plt.show()
print('The input is classified as digit: ', prediction)
