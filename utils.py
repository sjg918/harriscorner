

import torch
import hcd_on_gpu
import cv2
import numpy as np
import time

if __name__ == '__main__':    
    Limg = cv2.imread('left.png')
    newimg = cv2.imread('left.png')
    h, w, _ = Limg.shape
    Limg = cv2.cvtColor(Limg, cv2.COLOR_BGR2GRAY)
    Limg = torch.from_numpy(Limg).to(dtype=torch.float32, device='cuda:0')

    Rhost = hcd_on_gpu.hcd(Limg, w, h)
    mask = Rhost > 0.02 * Rhost.max()
    mask = mask.reshape(h, w)
    mask = mask.cpu().numpy()

    newimg[mask, 0] = 0
    newimg[mask, 1] = 0
    newimg[mask, 2] = 255
    cv2.imwrite('hi.png', newimg)
    print(Limg.shape)
    print(Rhost.sum())
    df=df
