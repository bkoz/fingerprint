import numpy
import json
import cv2

img_size = 96
datadir = '../data/fingerprint_real'

M_Right_index = cv2.imread(f'{datadir}/504__M_Right_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(M_Right_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)

req = {
    "inputs": [
      {
        "name": "conv2d_12_input",
        "shape": [1, 96, 96, 1],
        "datatype": "FP32",
        "data": img_resize.tolist()
      }
    ]
  }

with open('request-fingerprint.json', 'w') as convert_file:
     convert_file.write(json.dumps(req))
