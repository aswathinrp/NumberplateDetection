import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import torch
import pytesseract as pt
# from tensorflow.keras.preprocessing.image import load_img, from tensorflow.keras.utils import load_img
# from keras.preprocessing import load_img,img_to_array
from tensorflow.keras.utils import load_img,img_to_array
model = torch.hub.load('aswathinrp/yolov5', 'custom', path='best.pt', force_reload=True)

def obj_detection(path,filename):
    image = load_img(path)
    image = np.array(image,dtype=np.uint8)
    image1 = load_img(path,target_size=(224,224))
    image_arr_224 = img_to_array(image1)/255.0
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
   
    cods = model.predict(test_arr)
    denorm = np.array([w,w,h,h])
    cods = cods * denorm
    cods = cods.astype(np.int32)
    xmin,xmax,ymin,ymax = cods[0]
    pt1 = (xmin,ymin)
    pt2 = (xmax,ymax)
    print(pt1,pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename),image_bgr)
    return cods

def OCR(path,filename):
    img = np.array(load_img(path))
    cods = obj_detection(path,filename)
    xmin,xmax,ymin,ymax = cods[0]
    roi = img[ymin:ymax,xmin:xmax]
    roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
    text = pt.image_to_string(roi)
    print(text)
    return text
    