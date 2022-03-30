import model
import os
from PIL import Image
from skimage.color import rgb2ycbcr
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2, glob, time

#scribble size
points = 10

# initialize
l_drawing, r_drawing = False, False

def onChange(pos):
    pass

def draw_circle(event,x,y,flags,param):
    global l_drawing, r_drawing

    if event == cv2.EVENT_RBUTTONDOWN:
        r_drawing = True
        l_drawing = False

    elif event == cv2.EVENT_LBUTTONDOWN:
        l_drawing = True
        r_drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if l_drawing == True:
            cv2.circle(inputs, (x,y), 5, (0,0,255), -1)
            cv2.circle(scribble, (x,y), points, 1, -1)
        elif r_drawing == True:
            cv2.circle(inputs, (x,y), 5, (255,0,0), -1)
            cv2.circle(scribble, (x,y), points, -1, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        l_drawing = False
        cv2.circle(inputs, (x,y), 5, (0,0,255), -1)
        cv2.circle(scribble, (x,y), points, 1, -1)

    elif event == cv2.EVENT_RBUTTONUP:
        r_drawing = False
        cv2.circle(inputs, (x,y), 5, (255,0,0), -1)
        cv2.circle(scribble, (x,y), points, -1, -1)


lst = glob.glob('/home/ksko/Desktop/Low-light/Video/Real_Dataset/Dynamic_noGT/dynamic_raw_data_noGT_2exp/11-22-17.31.41_seq_grab_loop_0912_3stop_2exps_4s_wb/*.jpg')
lst.sort()

# load image
img = Image.open(lst[0])
img = np.asarray(img)
h, w, _ = img.shape


img = Image.open(lst[0]).resize((w//4, h//4))
img = np.asarray(img)

# rgb2y -> Tensor
ycbcr = rgb2ycbcr(img)
y = ycbcr[..., 0] / 255.
y = torch.from_numpy(y).float()
y = y[None, None].cuda()

# rgb -> Tensor
lowlight = torch.from_numpy(img).float()
lowlight = lowlight.permute(2,0,1)
lowlight = lowlight.cuda().unsqueeze(0) / 255.


IceNet = model.IceNet().cuda()
IceNet.load_state_dict(torch.load('model/icenet.pth'))

resume = True

inputs = img.copy() / 255.
scribble = np.zeros(inputs.shape[:2])

drawing = False

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('image', draw_circle)
cv2.createTrackbar("threshold", "image", 0, 100, onChange)
cv2.setTrackbarPos("threshold", "image", 60)
global_e = cv2.getTrackbarPos("threshold", "image") / 100.
# annotations
s = torch.from_numpy(scribble)[None, None].float().cuda()
eta = torch.Tensor([global_e]).float().cuda()
# feedforward
enhanced_image = IceNet(y, s, eta, lowlight)
output = enhanced_image[0].permute(1, 2, 0).cpu().detach().numpy()
cv2.imshow('image', np.concatenate([inputs, output], 1)[..., ::-1])

step = 0

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('image', draw_circle)
cv2.createTrackbar("threshold", "image", 0, 100, onChange)
cv2.setTrackbarPos("threshold", "image", 60)

while(1):
    step = np.clip(step, 0, len(lst)-1)

    # load image
    img = Image.open(lst[step]).resize((w//4, h//4))
    img = np.asarray(img)

    # rgb2y -> Tensor
    ycbcr = rgb2ycbcr(img)
    y = ycbcr[..., 0] / 255.
    y = torch.from_numpy(y).float()
    y = y[None, None].cuda()

    # rgb -> Tensor
    lowlight = torch.from_numpy(img).float()
    lowlight = lowlight.permute(2,0,1)
    lowlight = lowlight.cuda().unsqueeze(0) / 255.


    inputs = img.copy() / 255.
    scribble = np.zeros(inputs.shape[:2])
    
    global_e = cv2.getTrackbarPos("threshold", "image") / 100.
    # annotations
    s = torch.from_numpy(scribble)[None, None].float().cuda()
    eta = torch.Tensor([global_e]).float().cuda()
    # feedforward
    enhanced_image = IceNet(y, s, eta, lowlight)
    output = enhanced_image[0].permute(1, 2, 0).cpu().detach().numpy()

    cv2.imshow('image', np.concatenate([inputs, output], 1)[..., ::-1])
    step += 1
    k = cv2.waitKey(1) & 0xFF
    time.sleep(0.1)
    # To reset , push key "1"
    if k == 49:
        step = 0
    if k == 50:
        break
