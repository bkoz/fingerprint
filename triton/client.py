# import json
import numpy
import requests
import cv2

#
# Get the model server info.
#
host = "https://fingerprint-sandbox.apps.hou.edgelab.online"
url = f'{host}/v2'
r = requests.get(url)
print(f'REST GET response = {r}')
print(f'REST GET content = {r.content}')

def make_prediction(img:numpy.array, img_size:int = 96)-> requests:
  """
  Make a prediction from a single image file.
  """
  req2 = {
      "inputs": [
        {
          "name": "conv2d_12_input",
          "shape": [1, 96, 96, 1],
          "datatype": "FP32",
          "data": img.tolist()
        }
      ]
    }

  url = f'{host}/v2/models/fingerprint/infer'
  r = requests.post(url, json=req2)
  return r


#
# Make 4 single image predictions.
#
img_size = 96
datadir = '../data/fingerprint_real'
F_Left_index = cv2.imread(f'{datadir}/103__F_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(F_Left_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
r = make_prediction(img_resize, img_size)
print(f'REST inference response = {r}')
print(f'REST inference content = {r.content}')

F_Left_index = cv2.imread(f'{datadir}/275__F_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(F_Left_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
r = make_prediction(img_resize, img_size)
print(f'REST inference response = {r}')
print(f'REST inference content = {r.content}')

M_Right_index = cv2.imread(f'{datadir}/232__M_Right_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(M_Right_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
r = make_prediction(img_resize, img_size)
print(f'REST inference response = {r}')
print(f'REST inference content = {r.content}')

M_Right_index = cv2.imread(f'{datadir}/504__M_Right_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(M_Right_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
r = make_prediction(img_resize, img_size)
print(f'REST inference response = {r}')
print(f'REST inference content = {r.content}')
