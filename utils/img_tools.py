import cv2
import numpy
import os

def CropImg(img, x: int, y: int, width: int, height : int):
    img_crop = img[y:y+height, x:x+width, :]
    return img_crop

if __name__ == '__main__':
    print('img_tool.py is running as main program.')
    # print(os.getcwd())
    # im = cv2.imread('./test.jpg')
    # print(im.shape)
    # im_crop = CropImg(im, 0, 0, 100, 50)
    # cv2.imwrite('test_crop.jpg', im_crop)

