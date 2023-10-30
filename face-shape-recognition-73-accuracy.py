#!/usr/bin/env python
# coding: utf-8

# # **NOTE**
# # Val_accuracy has a maximum of 75% 
# (changing network architecture won't work);<br>
# Because (after data investigation) I found that many samples are misclassifed & some samples are corrupted (face recognition & preprocessing issues)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from PIL import ImageFile
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    horizontal_flip = True,
                                    brightness_range=(0.8,1.2)
                                    )

test_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    brightness_range=(0.8,1.2),
                                    horizontal_flip = True,
                                    )


training_set = train_datagen.flow_from_directory(
                                                 '../input/faceshape-processed/dataset/train',
                                                 target_size = (250,190),
                                                 batch_size = 64,
                                                 color_mode = 'grayscale',
                                                 shuffle = True,
                                                 class_mode = 'categorical'
                                                 )

test_set = test_datagen.flow_from_directory(
                                            '../input/faceshape-processed/dataset/test',
                                            target_size = (250,190),
                                            batch_size = 64,
                                            color_mode = 'grayscale',
                                            shuffle=True,
                                            class_mode = 'categorical'
                                            )


# In[7]:


# Build model

model = Sequential()
model.add(Conv2D(8, (7, 7), activation='selu', padding="valid", input_shape=(250,190,1)))
model.add(Conv2D(8, (5, 5), activation='selu', padding="valid"))
model.add(MaxPooling2D(pool_size=(5,5),padding="valid", strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(3,3),padding="valid", strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='selu'))
model.add(Dense(5, activation="softmax"))

model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[8]:


X_train, y_train = next(training_set)
X_test, y_test = next(test_set)


# In[9]:


# Visualize convolution processing
from tensorflow.keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[0].reshape(1,250,190,1))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*4.5,col_size*2.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
col_size = 4
row_size = 2
n_layers = 4
for layer_index in range(n_layers):#(len(model.layers)-10):
    display_activation(activations, col_size, row_size, layer_index)


# In[10]:


# Custom early stop
class ValAccEarlyStop(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ValAccEarlyStop, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


# In[11]:


# Training phase
history = model.fit(
                             training_set,
                             epochs = 50,
                             validation_data = test_set,
                             shuffle = True,
                            callbacks=[
                                 ValAccEarlyStop(0.73)
                             ]
                            )


# In[12]:


# Evaluate Model
scoreSeg = model.evaluate_generator(test_set)
print("Accuracy = ",scoreSeg[1])


# In[13]:


model.save("face-shape-recognizer.h5")


# In[14]:


# Visualize Loss & Accuracy

get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[15]:


# Make some predictions on training data

from tensorflow.keras.preprocessing import image

labels = list(training_set.class_indices)

for i in np.random.randint(X_train.shape[0], size=10):
    img = image.img_to_array(X_train[i])
    img = np.expand_dims(img, axis=0)
    images = np.vstack([img])
    y_pred = np.argmax(model.predict(img,verbose=0), axis=1)[0]
    y_true = np.argmax(y_train[i])
    plt.imshow(X_train[i])
    plt.title(f"Y true({labels[y_true]}) | Y pred ({labels[y_pred]})")
    plt.show()


# In[ ]:




