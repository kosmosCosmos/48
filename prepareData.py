import io
from io import BytesIO

import cv2
import insightface
import numpy as np
import requests
from PIL import Image
from annoy import AnnoyIndex

ctx_id = -1
nms = 0.4
f = 512
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed

r = requests.get("http://h5.snh48.com/resource/jsonp/allmembers.php?gid=00&_=1586492165883").json()
for i in r['rows'][:1]:
    res = requests.get("https://www.snh48.com/images/member/zp_" + i['sid'] + ".jpg")
    print(i['tname'], i['sid'], i['pinyin'])
    image = cv2.imdecode(np.asarray(bytearray(res.content), dtype="uint8"), cv2.IMREAD_COLOR)
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=ctx_id, nms=nms)
    faces = model.get(image)
    for idx, face in enumerate(faces):
        data = face.bbox.astype(np.int).flatten()
        if abs(data[3] - data[1]) < abs(data[2] - data[0]):
            data[3] = data[3] + ((abs(data[2] - data[0])) - abs((data[3] - data[1])))
        else:
            data[2] = data[2] + (abs((data[3]) - data[1]) - abs(data[2] - data[0]))
        img = Image.open(BytesIO(res.content))
        cropped = img.crop(tuple(data))  # (left, upper, right, lower)
        scale = 1.0 * cropped.size[0] / 112
        new_im = cropped.resize((round(cropped.size[0] / scale), round(cropped.size[1] / scale)), Image.ANTIALIAS)
        imgByteArr = io.BytesIO()
        new_im.save(imgByteArr, format='PNG')
        model = insightface.model_zoo.get_model('arcface_r100_v1')
        model.prepare(ctx_id=ctx_id)
        emb = model.get_embedding(
            cv2.imdecode(np.asarray(bytearray(imgByteArr.getvalue()), dtype="uint8"), cv2.IMREAD_COLOR))
        print(int(i['sid']))
        t.add_item(int(i['sid']), emb[0])

t.build(10)  # 10 trees
t.save('test.ann')
