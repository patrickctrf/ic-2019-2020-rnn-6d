# univariate lstm example
from random import uniform, randint

import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

x_input = []
y_input = []
for i in range(15):
    x_input.append(randint(0,10))
    y_input.append(array(x_input).sum())

x_input = array(x_input)
y_input = array(y_input)

x_input = x_input/y_input.max()
y_input = y_input/y_input.max()

x_input = x_input.reshape(x_input.shape[0], 1, 1)

# define model
model = Sequential()
model.add(LSTM(50, activation='tanh', batch_input_shape=(1, 1, 1), stateful=True))
model.add(Dense(1))
model.compile(optimizer='nadam', loss='mse')
# fit model
model.fit(x_input,  y_input, epochs=500, verbose=0,batch_size=1, shuffle=False)
# demonstrate prediction
# x_input = x_input[0:4]
print("x_input: ", x_input, '\n')
print("y_input: ", y_input, '\n')
yhat = model.predict(x_input, batch_size=1, verbose=0)
print("yhat: ", yhat, '\n')
