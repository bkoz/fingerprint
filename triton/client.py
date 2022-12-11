# import json
import numpy
import requests
import cv2
import ast

def make_prediction(img:numpy.array, img_size:int, host:str)-> requests:
  """
  Make a binary prediction from a single fingerprint image.
  Args:
   img - The request image array.
   img_size - The image height and width.
   host - The hostname[:port] of the model service.
  Returns: The model output prediction.

  Consider passing in the image filename vs. an array?
  """
  req2 = {
      "inputs": [
        {
          "name": "conv2d_12_input",
          "shape": [1, img_size, img_size, 1],
          "datatype": "FP32",
          "data": img.tolist()
        }
      ]
    }

  url = f'{host}/v2/models/fingerprint/infer'
  r = requests.post(url, json=req2)
  return r

#
# Get the model server info.
#
host = "http://fingerprint-sandbox.apps.hou.edgelab.online"
url = f'{host}/v2'
r = requests.get(url)
print(f'REST GET response = {r}')
print(f'REST GET content = {r.content.decode()}')
print()

#
# Make 4 single image predictions.
#
img_size = 96
datadir = '../data/fingerprint_real'
filenames = [\
  '103__F_Left_index_finger.BMP',\
    '275__F_Left_index_finger.BMP',
    '232__M_Right_index_finger.BMP',
    '504__M_Right_index_finger.BMP'
    ]
for filename in filenames:
  F_Left_index = cv2.imread(f'{datadir}/{filename}', cv2.IMREAD_GRAYSCALE)
  img_resize = cv2.resize(F_Left_index, (img_size, img_size))
  img_resize.resize(1, img_size, img_size, 1)
  r = make_prediction(img_resize, img_size, host)
  print(f'REST inference response = {r}')
  # print(f'REST inference content = {r.content}')
  p = ast.literal_eval(r.content.decode())
  print(f"Image = {filename}, Prediction = {p['outputs'][0]['data']}")