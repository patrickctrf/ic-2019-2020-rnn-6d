import numpy as np
import csv

# Funcao sera utilizada para encontrar os valores mais próximos ao alinhar o
# array de temporizacao nas inputs e no ground truth.
# Retorna o indice do elemento cujo valor eh mais proximo do valor passado.
def find_nearest(array, value):
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
    :return: X e Y, respectivamente, uma matriz de (samples, steps, features) para entrada na rede neural e uma matriz com
    (samples, features) de dimensao, sendo a resposta esperada para cada valor de saida a cada sample.
    """

    X = []
    Y = []

    for single_input, i in zip(raw_input, range(raw_input.shape[0])):
        if i + steps < raw_input.shape[0]:
            X.append(raw_input[i:i+steps])
            Y.append(raw_output[i+steps])

    return np.array(X), np.array(Y)


#===============================================================================

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

#===============================================================================

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

#===============================================================================

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

#===============================================================================

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

#===============================================================================

# Here you can check if both time stamps (from IMU and ground truth) are aligned
# and have the same size. Check the maximum deviation too.

# Each timestamp and their difference.
for i,j in zip(timestampList[0:100], timestampListGroundTruth[0:100]):
    print(i, '\t\t', j, '\t\t', i-j)

# Maximum deviation in total timestamp.
print(max((i-j).min(), (i-j).max(), key=abs))

#===============================================================================

# Concatenating input data.

# raw_input = np.concatenate([[timestampList], [accelX], [accelY], [accelZ], [gyroX], [gyroY], [gyroZ]], axis=1)

# (28730 amostras, 7 sensores)
raw_input = np.array([timestampList, accelX, accelY, accelZ, gyroX, gyroY, gyroZ]).T

raw_output = np.array([positionX, positionY, positionZ, quaternionW, quaternionX, quaternionY, quaternionZ]).T

X,Y = split_dataset(raw_input, raw_output, steps=10)

raw_input[0]
