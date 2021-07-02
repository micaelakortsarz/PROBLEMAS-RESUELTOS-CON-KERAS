import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import PIL.Image

def compute_loss(input_image, filter_index,model,layer_name):
    layer = model.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    activation = feature_extractor(input_image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

def gradient_ascent_step(img, filter_index, learning_rate,model,layer_name):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index,model,layer_name)
    grads = tape.gradient(loss, img)
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image(input_size):
 #   img = PIL.Image.open('/content/mica.jpg')
  #  img=img.resize((input_size,input_size))
   # img=tf.convert_to_tensor(np.array(img),dtype=tf.float32)
    #img=tf.reshape(img,(1,input_size,input_size,3))*0.1
    img =tf.random.uniform((1, input_size, input_size, 3))
    return (img - 0.5) * 0.25

def visualize_filter(filter_index,model,layer_name,input_size):
    iterations = 30
    learning_rate = 10.0
    img = initialize_image(input_size)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate,model,layer_name)
    img = deprocess_image(img[0].numpy())
    return loss, img

def deprocess_image(img):
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15
    img = img[25:-25, 25:-25, :]
    img += 0.5
    img = np.clip(img, 0, 1)
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

def display_N_filters(image_size,n_filters,n_rows,model,layer_name):
  all_imgs = []
  for filter_index in range(n_filters):
      loss, img = visualize_filter(filter_index,model,layer_name,input_size)
      all_imgs.append(img)
  margin = 2
  n = n_rows
  cropped_width = input_size - 25 * 2
  cropped_height = input_size - 25 * 2
  width = n * cropped_width + (n - 1) * margin
  height = n * cropped_height + (n - 1) * margin
  stitched_filters = np.zeros((width, height, 3))
  for i in range(n):
      for j in range(n):
          img = all_imgs[i * n + j]
          stitched_filters[
              (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
              (cropped_height + margin) * j : (cropped_height + margin) * j
              + cropped_height,
              :,
          ] = img
  keras.preprocessing.image.save_img("resnet50v2{0}.png".format(layer_name), stitched_filters)
  display(Image("resnet50v2{0}.png".format(layer_name)))




input_size=128
model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
layer_name = ['conv1_conv','conv2_block3_out', "conv3_block3_out", "conv4_block3_out", "conv5_block3_out"]
#layer_name = ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool','block1_conv2',"block2_conv2",'block3_conv3','block4_conv3','block5_conv3']
#model = keras.applications.VGG19(weights="imagenet", include_top=False)
#layer_name = ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool','block1_conv2',"block2_conv2",'block3_conv3','block4_conv3','block5_conv3']
#model = keras.applications.VGG16(weights="imagenet", include_top=False)
model.summary()
for l in layer_name:
  display_N_filters(input_size,16,4,model,l)
