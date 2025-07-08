#import dependencies and pretrained models
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-style-stylization-v1-256/2')

#preprocess image and load
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image-dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

content_image = load_img('profile.jfif')
style_image = load_img('monet.jpeg')

#visualise output
plt.imshow(np.squeeze(content_image))
plt.show

#stylize image
stylized_image = model(tf.constant(content_image), tf.constant(style_image))

plt.imshow(np.squeeze(stylized_image))
plt.show()

cv2.imwrite('generated_img.jpg', cv2.cvtcolor(np.squeeze(stylized_image),cv2.COLOR_BGR2RGB))


