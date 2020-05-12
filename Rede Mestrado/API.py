import numpy as np
import csv
import matplotlib.pyplot as plt

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
    idx = (np.abs(np.abs(array) - np.abs(value))).argmin()
    return idx

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

# Os dois arranjos não possuem o mesmo tamanho, sendo que o ground truth possui
# menos amostras (naturlmente um sistem de amostragem mais lento).
# Vamos INTERPOLAR os valores de ground truth para ficar com o mesmo tamanho das
# amostras da IMU.

positionX = np.interp(timestampList, timestampListGroundTruth, positionX)
positionY = np.interp(timestampList, timestampListGroundTruth, positionY)
positionZ = np.interp(timestampList, timestampListGroundTruth, positionZ)

quaternionW = np.interp(timestampList, timestampListGroundTruth, quaternionW)
quaternionX = np.interp(timestampList, timestampListGroundTruth, quaternionX)
quaternionY = np.interp(timestampList, timestampListGroundTruth, quaternionY)
quaternionZ = np.interp(timestampList, timestampListGroundTruth, quaternionZ)

# Esta linha precisa ficar por ultimo, porque o tempo de captura inicial do
# truth eh nossa referencia para alinhamento das demais grandezas.
timestampListGroundTruth = np.interp(timestampList, timestampListGroundTruth, timestampListGroundTruth)

# ==============================================================================

# (28730 amostras, 7 sensores)
raw_input = np.array([timestampList, accelX, accelY, accelZ, gyroX, gyroY, gyroZ]).T

# (28730 amostras de ground truth, 7 graus de liberdade)
raw_output = np.array([positionX, positionY, positionZ, quaternionW, quaternionX, quaternionY, quaternionZ]).T

plt.plot(timestampListGroundTruth, positionX, 'bs', timestampList, raw_output[:,0], 'r--')
plt.show()