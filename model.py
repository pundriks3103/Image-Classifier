#Importing the Libraries

import numpy as np
from keras.models import Sequential # To initialise the Neural Network
from keras.layers import Convolution2D # To add Convolutional Layers
from keras.layers import MaxPooling2D # For Pooling Step
from keras.layers import Flatten # For converting Pooled features to Matrix
from keras.layers import Dense, Dropout
from keras import optimizers

#Image Augmentation

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/content/Convolutional_Neural_Networks/dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/content/Convolutional_Neural_Networks/dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# Three Convolution Layer CNN Model (Modified) (using Dropout Regularization)
def cnn_model_modified():
  classifier = Sequential()
  classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2,2)))
  classifier.add(Dropout(0.2))
  classifier.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2,2)))
  classifier.add(Dropout(0.2))
  classifier.add(Convolution2D(128, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2,2)))
  classifier.add(Dropout(0.2))
  classifier.add(Flatten())
  classifier.add(Dense(units = 128, activation = 'relu'))
  classifier.add(Dropout(0.5))
  classifier.add(Dense(units = 1, activation = 'sigmoid'))
  # Compiling the CNN
  classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  return classifier

# Fitting the CNN Model_Modified to the images
model = cnn_model_modified()
accuracies = model.fit_generator(training_set, steps_per_epoch = 8000, epochs = 5, validation_data = test_set, validation_steps = 2000)

# Saving the Best Model

import pickle
pickle.dump(model, open('model.pkl','wb'))

# Making Sample Predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/content/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
actual_test_image = test_image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model3.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# Viewing the Actual Test Image
print("The actual image is :-")
actual_test_image

# Viewing the Prediction
print("The predicted image is a", prediction)
