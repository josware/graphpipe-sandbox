from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import requests

from graphpipe import remote

data = np.array(Image.open("test-imgs/mug227.png"))
data = data.reshape([1] + list(data.shape))
data = np.rollaxis(data, 3, 1).astype(np.float32)  # channels first
print(data.shape)

pred = remote.execute("http://127.0.0.1:9000", data)
print(np.argmax(pred, axis=1))
print("Expected 504 (Coffee mug), got: {}".format(np.argmax(pred, axis=1)))
print("Full quick start: https://oracle.github.io/graphpipe/#/guide/user-guide/quickstart")