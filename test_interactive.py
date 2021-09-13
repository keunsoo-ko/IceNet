import model
import os
from PIL import Image
from skimage.color import rgb2ycbcr
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='2'
points = 10

def onChange(pos):
    pass

def draw_circle(event,x,y,flags,param):
    global l_drawing, r_drawing, mode, points

    if event == cv2.EVENT_RBUTTONDOWN:
        r_drawing = True
        l_drawing = False

    elif event == cv2.EVENT_LBUTTONDOWN:
        l_drawing = True
        r_drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if l_drawing == True:
            cv2.circle(inputs, (x,y), 5, (0,0,255), -1)
            cv2.circle(maps, (x,y), 5, (0,0,255), -1)
            cv2.circle(scribble, (x,y), points, 1, -1)
        elif r_drawing == True:
            cv2.circle(inputs, (x,y), 5, (255,0,0), -1)
            cv2.circle(maps, (x,y), 5, (255,0,0), -1)
            cv2.circle(scribble, (x,y), points, -1, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        l_drawing = False
        cv2.circle(inputs, (x,y), 5, (0,0,255), -1)
        cv2.circle(maps, (x,y), 5, (0,0,255), -1)
        cv2.circle(scribble, (x,y), points, 1, -1)

    elif event == cv2.EVENT_RBUTTONUP:
        r_drawing = False
        cv2.circle(inputs, (x,y), 5, (255,0,0), -1)
        cv2.circle(maps, (x,y), 5, (255,0,0), -1)
        cv2.circle(scribble, (x,y), points, -1, -1)


img = Image.open('../Base/data/test_data/MEF/Lamp.png')
#img = img.resize((img.width // 4, img.height//4))
img = np.asarray(img)

ycbcr = rgb2ycbcr(img)
y = ycbcr[..., 0] / 255.

data_lowlight = torch.from_numpy(img).float()
data_lowlight = data_lowlight.permute(2,0,1)
data_lowlight = data_lowlight.cuda().unsqueeze(0) / 255.

y = torch.from_numpy(y).float()
y = y[None, None].cuda()

DCE_net = model.UIENet().cuda()
DCE_net.load_state_dict(torch.load('snapshots/Epoch40.pth'))

resume = True


while(resume):
    inputs = img.copy() / 255.
    scribble = np.zeros(inputs.shape[:2])
    maps = np.zeros_like(img)
    
    drawing = False
    mode = False
    ix, iy = -1, -1

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('image', draw_circle)
    cv2.createTrackbar("threshold", "image", 0, 120, onChange)
    cv2.setTrackbarPos("threshold", "image", 60)
    while(1):
        global_e = cv2.getTrackbarPos("threshold", "image") / 100.
        s = torch.from_numpy(scribble)[None, None].float().cuda()
        enhanced_Y, _ = DCE_net(y, s, torch.Tensor([global_e]).float().cuda())
        enhanced_image = torch.clip(enhanced_Y*(data_lowlight/y), 0, 1)
        output = enhanced_image[0].permute(1, 2, 0).cpu().detach().numpy()
        cv2.imshow('image', np.concatenate([inputs, output], 1)[..., ::-1])
        k = cv2.waitKey(1) & 0xFF
        if k == 49:
            resume = True
            break
        if k == 50:
            cv2.imwrite('results.png', output[..., ::-1]*255.)
        if k == 51:
            cv2.imwrite('inputs.png', img[..., ::-1])
        if k == 52:
            cv2.imwrite('scribble.png', maps[..., ::-1])
            points = 5
        if k == 53:
            points = 20
        if k == 27:
            resume = False
            break
    cv2.destroyAllWindows()
