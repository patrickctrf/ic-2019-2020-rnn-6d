import base64
import json
import time
from os import listdir
from os.path import isfile, join

import zmq


def producer():
    img_files_names = [f for f in listdir("/home/patrickctrf/Documentos/ORB_SLAM3/MH04/mav0/cam0/data") if isfile(join("/home/patrickctrf/Documentos/ORB_SLAM3/MH04/mav0/cam0/data", f))]
    img_files_names.sort()  # alphabetical order

    for i in range(len(img_files_names)):
        # setup socket
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PAIR)
        zmq_socket.bind("tcp://127.0.0.1:6009")

        # Read file content
        img_number = 1403638128445096960 + i
        f = open("/home/patrickctrf/Documentos/ORB_SLAM3/MH04/mav0/cam0/data/" +
                 img_files_names[i], 'rb')
        bytes = bytearray(f.read())

        # Encode to send
        strng = base64.b64encode(bytes)
        print("Sending file over")
        print("\n\nEncoded message size: ", len(bytes))  # 4194328 in my case
        print("\n\nEncoded message size: ", len(strng))  # 4194328 in my case
        zmq_socket.send(strng)
        pose_dict = json.loads(zmq_socket.recv_string())
        print("pose_dict: ", pose_dict)
        f.close()

        time.sleep(0.1)


producer()
