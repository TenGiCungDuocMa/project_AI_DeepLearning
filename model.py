from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(plt.imshow(x_test[0]))
# print(y_test[0])

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train[0])

# create model
model = Sequential()

#add model layer
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
model.summary()

model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=3)
# model.save("CSVT.h5")
# y_hat = model.predict(x_test[19:20])
# print(y_hat)
# y_label = np.argmax(y_hat, axis=1)
# print(y_label)
