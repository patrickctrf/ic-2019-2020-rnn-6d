import numpy as np
import csv
import matplotlib.pyplot as plt

from keras import Sequential, Input, Model
from keras.activations import exponential
from keras.engine.saving import model_from_json
from keras.layers import TimeDistributed, Dense, LSTM, Dropout, concatenate, LeakyReLU
from keras.optimizers import SGD
from keras.utils import plot_model


def find_nearest(array, value):
    """
    Funcao sera utilizada para encontrar os valores mais próximos ao alinhar o
    array de temporizacao nas inputs e no ground truth.
    Retorna o indice do elemento cujo valor eh mais proximo do valor passado.
    :param array: O array de valores onde buscaremos o valor mais proximo de "value".
    :param value: O valor aproximado a ser bucado no "array".
    :return: O indice do elemento do "array" cujo valor eh mais proximo de "value".
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def split_dataset(raw_input, raw_output, steps):
    """
    Esta funcao recebe a entrada e saida do dataset, ambas no formato (samples, features), onde samples eh igual para
    ambas, e retorna a entrada e saida para o treinamento, no formato (samples, steps, features) para a entrada X e
    (features) para a saida Y.

    :param raw_input: Dataset de entrada no formato (samples, features).
    :param raw_output: Dataset de saida no formato (samples, features).
    :param steps: Quantos timesteps tera cada entrada para o treinamento.
    :return: Xt e Yt, respectivamente, uma matriz de (samples, steps, features) para entrada na rede neural e uma matriz com
    (samples, features) de dimensao, sendo a resposta esperada para cada valor de saida a cada sample do tempo t. Xt_1 e
    Yt_1 sao as mesmas matrizes, mas os valores se referem ao tempo t-1.
    """

    Xt = []
    Yt = []

    Xt_1 = []
    Yt_1 = []

    for i in range(raw_input.shape[0]):

        # Como queremos Xt e Xt-1, precisamos começar do segundo ponto para haver um t-1 a ser coletado
        i = i + 1

        if i + steps < raw_input.shape[0]:
            Xt.append(raw_input[i:i + steps])

            # A parte da rede que prevê o step anterior recebe o ultimo
            # passo conhecido, por isso concatenamos parte da saida aqui.
            # Lembre que estamos concatenando a saida no temp "i" e tentando
            # calcular a saida em "i+steps" depois, entao a rede nao vai
            # fazer a funcao identidade aqui.
            Xt_1.append(np.concatenate([raw_input[i - 1:i - 1 + steps], np.tile(raw_output[i - 1], (steps, 1))], axis=1))

            # Estamos calculando quanto a posicao VARIOU durante esta amostra,
            # nao a localizacao absoluta atual.
            Yt.append(raw_output[i + steps])
            Yt_1.append(raw_output[i - 1 + steps])

    return np.array(Xt), np.array(Yt), np.array(Xt_1), np.array(Yt_1)

def repeteMatrizDataset(Xt, Xt_1, reps):
    """
    Como precisamos concatenar na ordem certa as diferentes entradas para a rede
    neural, esta funcao eh uma conveniencia que realiza esta concatenacao.
    :param Xt: ARRAY DE matrizes a serem concatenadas uma a uma.
    :param Xt_1: Segundo ARRAY DE matrizes a serem concatenadas uma a uma.
    :param reps: Quantas vezes repetir cada uma das matrizes durante a
    concatemação.
    :return Xtotal: Um ARRAY de matrizes ja concatenadas e repetidas conforme
    a quantidade passada como argumento.
    """
    Xtotal = []
    XtAux = []
    Xt_1Aux = []

    # Repetimos primeiramente as matrizes de entrada lateralmente, o numero de
    # vezes especificado em reps (repeticoes).
    for xt, xt_1 in zip(Xt, Xt_1):
        # Lembre que temos que repetir na segunda dimensao da matriz, a primera
        # nao repete e, por isso, o numero "1" no inicio da tupla abaixo.
        XtAux.append(np.tile(xt, (1,reps)))
        Xt_1Aux.append(np.tile(xt_1, (1, reps)))

    # Agora, basta concatenar as matrizes que ja foram repetidas.
    for xt, xt_1 in zip(XtAux, Xt_1Aux):
        Xtotal.append(np.concatenate([xt, xt_1], axis=1))

    return np.array(Xtotal)
# ===============================================================================

# Vamos ate o diretorio onde estao os dados de acelerometro e giroscopio da IMU
# e obtemos os dados dos mesmos no formato de listas.

# %cd /content/sample_data/dataset-room2_512_16/mav0/imu0

# Listas para receber os dados da IMU.
timestampList = []

accelX = []
accelY = []
accelZ = []

gyroX = []
gyroY = []
gyroZ = []

with open('/mnt/447acb07-56ed-4e48-842c-bd311a12cf0a/Downloads/dataset-room2_512_16/mav0/imu0/data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            timestampList.append(row[0])
            gyroX.append(row[1])
            gyroY.append(row[2])
            gyroZ.append(row[3])
            accelX.append(row[4])
            accelY.append(row[5])
            accelZ.append(row[6])

            line_count += 1
    print(f'Processed {line_count} lines.')

# ===============================================================================

# Vamos ate o diretorio onde estao os dados de ground truth e obtemos os dados
# de translacao nos eixos X,Y e Z, bem como os dados de rotacao na notacao de
# quaternion com eixos w, x, y e z (Tem a biblioteca TF para converter de volta
# a angulos de Euler).

# %cd /content/sample_data/dataset-room2_512_16/mav0/mocap0/


# Listas para receber os dados de ground truth.
timestampListGroundTruth = []

positionX = []
positionY = []
positionZ = []

quaternionW = []
quaternionX = []
quaternionY = []
quaternionZ = []

with open('/mnt/447acb07-56ed-4e48-842c-bd311a12cf0a/Downloads/dataset-room2_512_16/mav0/mocap0/data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            timestampListGroundTruth.append(row[0])
            positionX.append(row[1])
            positionY.append(row[2])
            positionZ.append(row[3])
            quaternionW.append(row[4])
            quaternionX.append(row[5])
            quaternionY.append(row[6])
            quaternionZ.append(row[7])

            line_count += 1
    print(f'Processed {line_count} lines.')

# ===============================================================================

# It's better to work with numpy arrays than lists.

# Input data.
timestampList = np.int64(timestampList)

accelX = np.float64(accelX)
accelY = np.float64(accelY)
accelZ = np.float64(accelZ)

gyroX = np.float64(gyroX)
gyroY = np.float64(gyroY)
gyroZ = np.float64(gyroZ)

# Ground Truth data.
timestampListGroundTruth = np.int64(timestampListGroundTruth)

positionX = np.float64(positionX)
positionY = np.float64(positionY)
positionZ = np.float64(positionZ)

quaternionW = np.float64(quaternionW)
quaternionX = np.float64(quaternionX)
quaternionY = np.float64(quaternionY)
quaternionZ = np.float64(quaternionZ)

# IMU data is not aligned to ground truth yet, they need to start togheter.

# We don't know which one started recording first. The biggest index from 
# "find_nearest" will tell us that.
idxIMU = find_nearest(timestampList, timestampListGroundTruth[0])
idxGndTruth = find_nearest(timestampListGroundTruth, timestampList[0])

# If we need to cut off initial readings from IMU.
if idxIMU > idxGndTruth:
    timestampList = timestampList[idxIMU:]

    accelX = accelX[idxIMU:]
    accelY = accelY[idxIMU:]
    accelZ = accelZ[idxIMU:]

    gyroX = gyroX[idxIMU:]
    gyroY = gyroY[idxIMU:]
    gyroZ = gyroZ[idxIMU:]

elif idxIMU < idxGndTruth:
    timestampListGroundTruth = timestampListGroundTruth[idxGndTruth:]

    positionX = positionX[idxGndTruth:]
    positionY = positionY[idxGndTruth:]
    positionZ = positionZ[idxGndTruth:]

    quaternionW = quaternionW[idxGndTruth:]
    quaternionX = quaternionX[idxGndTruth:]
    quaternionY = quaternionY[idxGndTruth:]
    quaternionZ = quaternionZ[idxGndTruth:]

# If they are the same, no need to cut anything.

# ===============================================================================

# Regularizando/Alinhando o TAMANHO das listas de input e ground truth do dataset (Precisam ter o mesmo tamanho).
# Obs: Nao sao mais listas do python, sao arrays de float no numpy.

novoTimestampListGroundTruth = []

novoPositionX = []
novoPositionY = []
novoPositionZ = []

novoQuaternionW = []
novoQuaternionX = []
novoQuaternionY = []
novoQuaternionZ = []

for timeElement, i in zip(timestampList, range(len(timestampList))):
    idx = find_nearest(timestampListGroundTruth, timeElement)

    novoTimestampListGroundTruth.append(timestampListGroundTruth[idx])

    novoPositionX.append(positionX[idx])
    novoPositionY.append(positionY[idx])
    novoPositionZ.append(positionZ[idx])

    novoQuaternionW.append(quaternionW[idx])
    novoQuaternionX.append(quaternionX[idx])
    novoQuaternionY.append(quaternionY[idx])
    novoQuaternionZ.append(quaternionZ[idx])

timestampListGroundTruth = np.int64(novoTimestampListGroundTruth)

positionX = np.float64(novoPositionX)
positionY = np.float64(novoPositionY)
positionZ = np.float64(novoPositionZ)

quaternionW = np.float64(novoQuaternionW)
quaternionX = np.float64(novoQuaternionX)
quaternionY = np.float64(novoQuaternionY)
quaternionZ = np.float64(novoQuaternionZ)

# ===============================================================================

# Here you can check if both time stamps (from IMU and ground truth) are aligned
# and have the same size. Check the maximum deviation too.

# Each timestamp and their difference.
for i, j in zip(timestampList[0:100], timestampListGroundTruth[0:100]):
    print(i, '\t\t', j, '\t\t', i - j)

# Maximum deviation in total timestamp.
print(max((i - j).min(), (i - j).max(), key=abs))

# ===============================================================================

# Normalizando dados antes de inserir como entrada. MUITO IMPORTANTE.

accelX = accelX / np.max(accelX)
accelY = accelY / np.max(accelY)
accelZ = accelZ / np.max(accelZ)

gyroX = gyroX / np.max(gyroX)
gyroY = gyroY / np.max(gyroY)
gyroZ = gyroZ / np.max(gyroZ)

timestampList = timestampList - np.min(timestampList)
timestampList = timestampList / np.max(timestampList)

# Concatenating input data.

# (28730 amostras, 7 sensores)
raw_input = np.array([timestampList, accelX, accelY, accelZ, gyroX, gyroY, gyroZ]).T

# (28730 amostras de ground truth, 7 graus de liberdade)
raw_output = np.array([positionX, positionY, positionZ, quaternionW, quaternionX, quaternionY, quaternionZ]).T

# Entradas do dataset (X) e suas respectivas saidas (Y).
n_steps = 30
Xt, Yt, Xt_1, Yt_1 = split_dataset(raw_input, raw_output, steps=n_steps)

# ===============================================================================

# neuroniosCamada1 = 50
# neuroniosCamada2 = 50
# neuroniosCamada3 = 50
# neuroniosCamada4 = 50
#
# input4 = Input(shape=(Xt_1.shape[1], Xt_1.shape[2]))
# output4 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt_1.shape[1], Xt_1.shape[2]))(
#     input4)
# output4 = Dropout(0.5)(output4)
# output4 = TimeDistributed(Dense(neuroniosCamada2, activation="tanh"))(output4)
# output4 = Dropout(0.5)(output4)
# output4 = TimeDistributed(Dense(neuroniosCamada3, activation="tanh"))(output4)
# output4 = Dropout(0.5)(output4)
# output4 = TimeDistributed(Dense(neuroniosCamada4, activation="tanh"))(output4)
# output4 = Dropout(0.5)(output4)
#
# input3 = Input(shape=(Xt_1.shape[1], Xt_1.shape[2]))
# output3 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt_1.shape[1], Xt_1.shape[2]))(
#     input3)
# output3 = Dropout(0.3)(output3)
# output3 = TimeDistributed(Dense(neuroniosCamada2, activation="tanh"))(output3)
# output3 = Dropout(0.3)(output3)
# output3 = TimeDistributed(Dense(neuroniosCamada3, activation="tanh"))(output3)
# output3 = Dropout(0.3)(output3)
#
# input2 = Input(shape=(Xt_1.shape[1], Xt_1.shape[2]))
# output2 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt_1.shape[1], Xt_1.shape[2]))(
#     input2)
# output2 = Dropout(0.5)(output2)
# output2 = TimeDistributed(Dense(neuroniosCamada2, activation="tanh"))(output2)
# output2 = Dropout(0.5)(output2)
#
# input1 = Input(shape=(Xt_1.shape[1], Xt_1.shape[2]))
# output1 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt_1.shape[1], Xt_1.shape[2]))(
#     input1)
# output1 = Dropout(0.5)(output1)
#
# merged = concatenate([output4, output3, output2, output1])
#
# merged_0 = LSTM(50, activation="tanh", return_sequences=True)(merged)
# merged = LSTM(50, activation="tanh")(merged_0)
# merged = Dense(50, activation="tanh")(merged)
# merged = Dropout(0.5)(merged)
# merged = Dense(Yt_1.shape[1])(merged)
#
#
#
# # Segunda etapa
# input4_2 = Input(shape=(Xt.shape[1], Xt.shape[2]))
# output4_2 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt.shape[1], Xt.shape[2]))(
#     input4_2)
# output4_2 = Dropout(0.5)(output4_2)
# output4_2 = TimeDistributed(Dense(neuroniosCamada2, activation="tanh"))(output4_2)
# output4_2 = Dropout(0.5)(output4_2)
# output4_2 = TimeDistributed(Dense(neuroniosCamada3, activation="tanh"))(output4_2)
# output4_2 = Dropout(0.5)(output4_2)
# output4_2 = TimeDistributed(Dense(neuroniosCamada4, activation="tanh"))(output4_2)
# output4_2 = Dropout(0.5)(output4_2)
#
# input3_2 = Input(shape=(Xt.shape[1], Xt.shape[2]))
# output3_2 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt.shape[1], Xt.shape[2]))(
#     input3_2)
# output3_2 = Dropout(0.3)(output3_2)
# output3_2 = TimeDistributed(Dense(neuroniosCamada2, activation="tanh"))(output3_2)
# output3_2 = Dropout(0.3)(output3_2)
# output3_2 = TimeDistributed(Dense(neuroniosCamada3, activation="tanh"))(output3_2)
# output3_2 = Dropout(0.3)(output3_2)
#
# input2_2 = Input(shape=(Xt.shape[1], Xt.shape[2]))
# output2_2 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt.shape[1], Xt.shape[2]))(
#     input2_2)
# output2_2 = Dropout(0.5)(output2_2)
# output2_2 = TimeDistributed(Dense(neuroniosCamada2, activation="tanh"))(output2_2)
# output2_2 = Dropout(0.5)(output2_2)
#
# input1_2 = Input(shape=(Xt.shape[1], Xt.shape[2]))
# output1_2 = TimeDistributed(Dense(neuroniosCamada1, activation="tanh"), input_shape=(Xt.shape[1], Xt.shape[2]))(
#     input1_2)
# output1_2 = Dropout(0.5)(output1_2)
#
# merged_2 = concatenate([output4_2, output3_2, output2_2, output1_2, merged_0])
#
# merged_2 = LSTM(50, activation="tanh")(merged_2)
# merged_2 = Dense(50, activation="tanh")(merged_2)
# merged_2 = concatenate([merged_2, merged])
# merged_2 = Dense(50, activation="tanh")(merged_2)
# merged_2 = Dropout(0.5)(merged_2)
# merged_2 = Dense(Yt.shape[1])(merged_2)
#
# model = Model(inputs=[input4_2, input3_2, input2_2, input1_2, input4, input3, input2, input1],
#               outputs=[merged_2, merged])
#
# model.compile(optimizer="nadam", loss='mean_squared_error')
#
# # Print model
# plot_model(model, to_file='2etapas.png', show_shapes=True)
#
# # Xfinal = repeteMatrizDataset(Xt, Xt_1, 4)
#
# # fit model
# model.fit([Xt, Xt, Xt, Xt, Xt_1, Xt_1, Xt_1, Xt_1], [Yt, Yt_1], epochs=5)

#==================================================================================

# load json and create model
json_file = open('model2etapas.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model2etapas.h5")
print("Loaded model from disk")

model = loaded_model
# # fit model
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.5, nesterov=True), loss='mean_squared_error')
# model.fit(Xt, Yt, epochs=250)

# # Saving model
#
# # %cd /content/sample_data
#
# model_json = model.to_json()
# json_file = open("model2etapas.json", "w")
# json_file.write(model_json)
# json_file.close()
# model.save_weights("model2etapas.h5")
# print("Model saved to disk")

# Print model
plot_model(model, to_file='model2etapas.png', show_shapes=True)

Ycalc = []
Ytrajetoria = []

# for i in range(len(Y) - n_steps):
#     print("Gerando predições", i)
#
#     #model.predict(X[i].reshape(1, Xt.shape[1], Xt.shape[2]))

# Ycalc = model.predict([Xt, Xt, Xt, Xt, Xt_1, Xt_1, Xt_1, Xt_1], verbose=1)


# A posição inicial do robo eh conhecida, entao passamos apenas ela para o sistema.
Ytrajetoria.append(Yt_1[0])
# Iterando sobre cada SAMPLE.
for i in range(Xt.shape[0]):
    print("Executando iteração:", i)
    # Iterando sobre cada STEP
    for j in range(Xt.shape[1]):
        # Iterando sobre cada SAIDA
        for k in range(Yt.shape[1]):
            Xt_1[i][j][k] = Ytrajetoria[i][k]
    # Depois de preencher cada sample por completo, faz o 'predict' dela.
    Ytrajetoria.append(model.predict([Xt[i].reshape(1, Xt.shape[1], Xt.shape[2]), Xt[i].reshape(1, Xt.shape[1], Xt.shape[2]), Xt[i].reshape(1, Xt.shape[1], Xt.shape[2]), Xt[i].reshape(1, Xt.shape[1], Xt.shape[2]), Xt_1[i].reshape(1, Xt_1.shape[1], Xt_1.shape[2]), Xt_1[i].reshape(1, Xt_1.shape[1], Xt_1.shape[2]), Xt_1[i].reshape(1, Xt_1.shape[1], Xt_1.shape[2]), Xt_1[i].reshape(1, Xt_1.shape[1], Xt_1.shape[2]).reshape(1, Xt_1.shape[1], Xt_1.shape[2])], verbose=1)[0][0])

# plt.plot(timestampList[31:], [i[0] for i in Yt], timestampList[31:], [i[0] for i in Ycalc[0]])
plt.plot(timestampList[31:], [i[0] for i in Yt], timestampList[31:], [i[0] for i in Ytrajetoria[:-1]])
plt.show()

# # demonstrate prediction
# x_input = np.array([70, 80, 90])
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
