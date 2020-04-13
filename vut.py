import os

import cv2


def func(img, pic_path, pic, number):
    # 把图片转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2]
    dark_sum = 0  # 偏暗的像素 初始化为0个
    dark_prop = 0  # 偏暗像素所占比例初始化为0
    piexs_sum = r * c  # 整个弧度图的像素个数为r*c

    try:
        os.mkdir("lightDir" + str(number))
    except OSError:
        pass
    # 遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum < 40:  # 人为设置的超参数,表示0~39的灰度值为暗
                dark_sum += 1
    dark_prop = dark_sum / (piexs_sum)
    if dark_prop >= 0.3:  # 人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
        cv2.imwrite("DarkPicDir/" + pic, img)  # 把被认为黑暗的图片保存
        return True
    else:
        cv2.imwrite("lightDir" + str(number) + "/" + pic, img)  # 把被认为黑暗的图片保存
        return False


# 读取给定目录的所有图片
def readAllPictures(pics_path):
    idx = 0
    number = 0
    if not os.path.exists(pics_path):
        print("路径错误，路径不存在！")
        return
    allPics = []
    pics = os.listdir(pics_path)
    for pic in pics:
        pic_path = os.path.join(pics_path, pic)
        if os.path.isfile(pic_path):
            allPics.append(pic_path)
            img = cv2.imread(pic_path)
            res = func(img, pic_path, pic, number)
            if res:
                idx = idx + 1
            else:
                idx = 0
            if idx == 8:
                number = number + 1
                idx = 0
            if res:
                print(pic, number, idx, "dark")
            else:
                print(pic, number, idx, "light")
    return allPics


if __name__ == '__main__':
    pics_path = "frames2"  # 获取所给图片目录
    allPics = readAllPictures(pics_path)
