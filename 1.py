# -*- coding: utf-8 -*-
import os

import cv2  ##加载OpenCV模块


def video2frames(pathIn='',
                 pathOut='',
                 extract_time_points=None,
                 output_prefix='frame', ):
    cap = cv2.VideoCapture(pathIn)  ##打开视频文件
    try:
        os.mkdir(pathOut)
    except OSError:
        pass
    success = True
    count = 0
    while success and count < len(extract_time_points):
        cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * extract_time_points[count]))
        success, image = cap.read()
        if success:
            print('Write a new frame: {}, {}th'.format(success, count + 1))
            cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)),
                        image)  # save frame as JPEG file
            count = count + 1


pathIn = '4.flv'
cap = cv2.VideoCapture(pathIn)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
dur = n_frames / fps
pathOut = './frames2'
points = []
for i in range(0, round(dur), 5):
    points.append(i)

video2frames(pathIn, pathOut, extract_time_points=tuple(points))
