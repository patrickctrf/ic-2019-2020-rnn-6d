# The RNN logic is written using Keras's functional API.
# Which means we use Keras layers instead of theano/tensorflow ops
from keras.layers import *
from keras.models import *
from recurrentshop import *

x_t = Input(shape=(5,)) # The input to the RNN at time t
h_tm1 = Input(shape=(10,))  # Previous hidden state

# Compute new hidden state
h_t = add([Dense(10)(x_t), Dense(10, use_bias=False)(h_tm1)])

# tanh activation
h_t = Activation('tanh')(h_t)

# Build the RNN
# RecurrentModel is a standard Keras `Recurrent` layer.
# RecurrentModel also accepts arguments such as unroll, return_sequences etc
rnn = RecurrentModel(input=x_t, initial_states=[h_tm1], output=h_t, final_states=[h_t])
# return_sequences is False by default
# so it only returns the last h_t state

# Build a Keras Model using our RNN layer
# input dimensions are (Time_steps, Depth)
x = Input(shape=(7,5))
y = rnn(x)
model = Model(x, y)

# Run the RNN over a random sequence
# Don't forget the batch shape when calling the model!
out = model.predict(np.random.random((1, 7, 5)))
print(out.shape)#->(1,10)


# to get one output per input sequence element, set return_sequences=True
rnn2 = RecurrentModel(input=x_t, initial_states=[h_tm1], output=h_t, final_states=[h_t],return_sequences=True)

# Time_steps can also be None to allow variable Sequence Length
# Note that this is not compatible with unroll=True
x = Input(shape=(None ,5))
y = rnn2(x)
model2 = Model(x, y)

out2 = model2.predict(np.random.random((1, 7, 5)))
print(out2.shape)#->(1,7,10)