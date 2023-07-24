import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from tabulate import tabulate

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#normalize greyscale from 0-255 to 0-1 then reshape the arrays to fit the CNN input
x_train = np.array(keras.utils.normalize(x_train, axis =1)).reshape(-1,28, 28, 1)
x_test = np.array(keras.utils.normalize(x_test, axis =1)).reshape(-1,28, 28, 1)

#Creating the CNN
model = Sequential()
#1st convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#2nd convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#3rd convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#1st fully connected layer
model.add(Flatten())
model.add(Dense(64, activation='relu'))

#2nd fully connected layer
model.add(Flatten())
model.add(Dense(32, activation='relu'))

#3rd fully connected layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#complile The CNN
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

#Train and validate the CNN
epochs = 3
history = model.fit(x_train, y_train, epochs= epochs, validation_split=0.3) #30% of training dataset is used for validation

print("The model has successfully trained")

#Evaluating the model and formatting the results in a table
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
data = [
    ["Training"  , round(history.history['loss'][epochs-1]*100     , 2)     , round(history.history['accuracy'][epochs-1]*100     , 2) ],
    ["Validation", round(history.history['val_loss'][epochs-1]*100 , 2)     , round(history.history['val_accuracy'][epochs-1]*100 , 2) ],
    ["Testing"   , round(loss*100                                  , 2)     , round(accuracy*100                                  , 2) ]
]

print (tabulate(data, headers=["Stage", "Loss / %", "Accuracy / %"]))

#saving the model
# no need to rerun the CNN again model is saved
model.save('handwritten.model')

'''
#I have this commented out as I have a GUI instead

model = keras.models.load_model('handwritten.model')
import os
import cv2
import matplotlib.pyplot as plt
i=0
while os.path.isfile(f"test/{i}.png"):
    img = cv2.imread(f"test/{i}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("This image is probaly a", np.argmax(prediction))
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()
    i+=1
#this yeilds a 90% accuracy
'''