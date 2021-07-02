
import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.datasets import cifar10, mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def Plot_Resultados(epocas,loss,ac,ac_test,label):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)

        ax1 = plt.subplot(311)
        plt.plot(epocas,loss, label=label)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        ax2 = plt.subplot(312)
        plt.plot(epocas,ac, label=label)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()
        ax3 = plt.subplot(313)
        plt.plot(epocas, ac_test, label=label)
        plt.ylabel(r'Accuracy test')
        plt.xlabel(r'Epocas transcurridas')

        plt.legend()
        plt.tight_layout()

plt.figure(figsize=(20, 8), constrained_layout=True)
epochs = 10
epocas=np.linspace(1,epochs,epochs)


input_shape = (32, 32, 3)
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
X_train_resized = []
for img in X_train:
  X_train_resized.append(np.resize(img, input_shape) / 255)
X_train_resized = np.array(X_train_resized)
X_test_resized = []
for img in X_test:
  X_test_resized.append(np.resize(img, input_shape) / 255)
X_test_resized = np.array(X_test_resized)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.summary()

for layer in base_model.layers:
  layer.trainable = False
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax', name='predictions')(x)
model = Model(base_model.input, x)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
batch_size = 256

h=model.fit(X_train_resized, Y_train,
          batch_size=batch_size,
          validation_data=(X_test_resized, Y_test),
          epochs=epochs)
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'VGG16 - MNIST')
scores = model.evaluate(X_test_resized, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Fine tuning

for layer in base_model.layers:
  if layer.name == 'block5_conv1':
    break
  layer.trainable = False

model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
epochs = 10
batch_size = 256
h=model.fit(X_train_resized, Y_train,
          batch_size=batch_size,
          validation_data=(X_test_resized, Y_test),
          epochs=epochs)
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'VGG16 (Fine Tuning) - MNIST')
scores = model.evaluate(X_test_resized, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Sin transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.summary()

for layer in base_model.layers:
  layer.trainable = True
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax', name='predictions')(x)
model = Model(base_model.input, x)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
batch_size = 256

h=model.fit(X_train_resized, Y_train,
          batch_size=batch_size,
          validation_data=(X_test_resized, Y_test),
          epochs=epochs)
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'VGG16 (Sin transferencia de conocimiento) - MNIST')
scores = model.evaluate(X_test_resized, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
plt.savefig('p3mnist.pdf')
plt.show()