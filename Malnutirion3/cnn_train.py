# -*- coding: utf-8 -*-


from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
import cv2
import matplotlib.pyplot as plt


#init the model
model= Sequential()

#add conv layers and pooling layers 
model.add(Convolution2D(32,3,3, input_shape=(400,400,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3, input_shape=(400,400,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5)) #to reduce overfitting

model.add(Flatten())

#Now two hidden(dense) layers:
model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.5))#again for regularization

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))


model.add(Dropout(0.5))#last one lol

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

#output layer
model.add(Dense(output_dim = 4, activation = 'sigmoid'))


#Now copile it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Now generate training and test sets from folders

train_datagen=ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False
                                 )

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory("Datasets/training_set",
                                               target_size = (400,400),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory("Datasets/test_set",
                                               target_size = (400,400),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')






#finally, start training
hiss=model.fit_generator(training_set,
                         samples_per_epoch = 520,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 320)



plt.figure(figsize=(15,7))

plt.subplot(1,2,1)
plt.plot(hiss.history['acc'], label='train')
plt.plot(hiss.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(hiss.history['loss'], label='train')
plt.plot(hiss.history['val_loss'], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#saving the weights
model.summary()

model.save_weights("weights.hdf5",overwrite=True)

#saving the model itself in json format:
model_json = model.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")






