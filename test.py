import numpy as np
import cv2

# resp = open("test.jpg", "rb")
# b=resp.read()
#
# image = np.asarray(bytearray(b), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)

import io
from PIL import Image

img = Image.open("test.jpg", mode='r')

imgByteArr = io.BytesIO()
img.save(imgByteArr, format='PNG')
imgByteArr = imgByteArr.getvalue()
image = np.asarray(bytearray(imgByteArr), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)