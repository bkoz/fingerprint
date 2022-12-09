#
# BEGIN: Create a request json file. Make this a separate source file.
#
import numpy
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

img_resize = numpy.array([[[[1], [2], [3]]]])

encodedNumpyData = json.dumps(img_resize, cls=NumpyArrayEncoder)

print("serialize NumPy array into JSON and write into a file")
with open("fingerprint.json", "w") as write_file:
    json.dump(img_resize, write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")

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

#
# END: Create a request json file.
#
