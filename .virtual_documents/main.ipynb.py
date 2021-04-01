import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.nn import relu, sigmoid
import matplotlib.pyplot as plt


x = [ [ [i + j] for i in range(5)] for j in range(100_000) ]
y = [i+5 for i in range(100_000)]


x = np.array(x)
y = np.array(y)
x = x / 100_000
y = y / 100_000
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = Sequential()

model.add(LSTM(64, input_shape=(5,1), activation=relu, return_sequences=True))

model.add(LSTM(128, activation=relu, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(64, activation=relu, return_sequences=True))
model.add(BatchNormalization())

model.add(LSTM(64, activation=relu, return_sequences=False))
model.add(Dropout(0.2))

model.add(Flatten())dsi
model.add(Dense(1, activation=sigmoid))


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))


pred = model.predict(x_test[:20])


plt.scatter(range(20), pred, color="r")
plt.scatter(range(20), y_test[:20], color="g")
plt.show()



