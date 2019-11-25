from keras import Input, Model
from keras.layers import TimeDistributed, Dropout, Dense, LSTM, concatenate

neuroniosCamada1 = 50
neuroniosCamada2 = 50
neuroniosCamada3 = 50
neuroniosCamada4 = 50

input4 = Input(shape=(X.shape[1], X.shape[2]))
output4 = TimeDistributed(Dense(neuroniosCamada1, activation='tanh'), input_shape=(X.shape[1], X.shape[2]))(input4)
output4 = Dropout(0.5)(output4)
output4 = TimeDistributed(Dense(neuroniosCamada2, activation='tanh'))(output4)
output4 = Dropout(0.5)(output4)
output4 = TimeDistributed(Dense(neuroniosCamada3, activation='tanh'))(output4)
output4 = Dropout(0.5)(output4)
output4 = TimeDistributed(Dense(neuroniosCamada4, activation='tanh'))(output4)
output4 = Dropout(0.5)(output4)

input3 = Input(shape=(X.shape[1], X.shape[2]))
output3 = TimeDistributed(Dense(neuroniosCamada1, activation='tanh'), input_shape=(X.shape[1], X.shape[2]))(input3)
output3 = Dropout(0.5)(output3)
output3 = TimeDistributed(Dense(neuroniosCamada2, activation='tanh'))(output3)
output3 = Dropout(0.5)(output3)
output3 = TimeDistributed(Dense(neuroniosCamada3, activation='tanh'))(output3)
output3 = Dropout(0.5)(output3)

input2 = Input(shape=(X.shape[1], X.shape[2]))
output2 = TimeDistributed(Dense(neuroniosCamada1, activation='tanh'), input_shape=(X.shape[1], X.shape[2]))(input2)
output2 = Dropout(0.5)(output2)
output2 = TimeDistributed(Dense(neuroniosCamada2, activation='tanh'))(output2)
output2 = Dropout(0.5)(output2)

input1 = Input(shape=(X.shape[1], X.shape[2]))
output1 = TimeDistributed(Dense(neuroniosCamada1, activation='tanh'), input_shape=(X.shape[1], X.shape[2]))(input1)
output1 = Dropout(0.5)(output1)

merged = concatenate([output4, output3, output2, output1])

merged = LSTM(50, activation='tanh')(merged)
merged = Dense(50, activation='tanh')(merged)
merged = Dropout(0.5)(merged)
merged = Dense(X.shape[2])(merged)

model = Model(inputs=[input4, input3, input2, input1], outputs=[merged])

model.compile(optimizer=SGD(lr=0.0001, momentum=0.8, nesterov=True), loss='mean_squared_error')

# Print model
plot_model(model, to_file='demoNovo.png',show_shapes=True)

# # fit model
# model.fit([X] * 4, Y, epochs=5)
