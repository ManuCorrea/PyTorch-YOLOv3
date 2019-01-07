from __future__ import division

from models import *
from utils.utils import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2
from pathlib import Path

from skimage.transform import resize

IMG_PATH = 'C:/Users/yo/Desktop/SelfDriving/joystickrecolDataTraining/test.bmp'

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='E:/checkpoints/96 ult.weights', help='path to weights file')
parser.add_argument('--gta_image', type=str, default=IMG_PATH,
                    help='location of file wanted to process')
parser.add_argument('--class_path', type=str, default='E:/YOLOv3/gta.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)


# Function to pass from img path to return a pytorch tensor of that image
# and a numpy's image
def transform_img(img_path):
    img = np.array(Image.open(img_path))
    img_size=416
    img_shape = (img_size, img_size)
    # Handles images with less than three channels
    img = np.array(Image.open(img_path))
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
    # Resize and normalize
    input_img = resize(input_img, (*img_shape, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()

    return input_img, img

# This function draws squares over the detected objets
def plotDetect(img, detections):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if detections is not None:
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x
        for detection in detections:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:

                print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                y2 = y1 + box_h
                x2 = x1 + box_w
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,classes[int(cls_pred)],(x1,y1), font, 1,(255,255,255),1,cv2.LINE_AA)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0))

        return img
    else:
        return img

cuda = torch.cuda.is_available() and opt.use_cuda

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
while True:
    #This try is because np.array(Image.open(img_path)) sometimes doesn't get an image an raises an error
    try: 
        input_imgs, img = transform_img(opt.gta_image)
        #Add batch dimension
        input_imgs = input_imgs.unsqueeze(0)
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print ('\t+ Inference Time: %s' % (inference_time))
        if detections[0] is not None:
            img = plotDetect(img, detections)
        cv2.imshow('window2', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    except:
        raise








