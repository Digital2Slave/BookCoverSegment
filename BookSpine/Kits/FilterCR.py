#-*- coding: UTF-8 -*-

import cv2
import numpy as np

def line_size(x1, y1, x2, y2):
    '''
    Line_size
    :param x1:pt1x
    :param y1:pt1y
    :param x2:pt2x
    :param y2:pt2y
    :return: distance between pt1 and pt2
    '''
    return np.sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)

def FilterCRS(CRimg, pList):
    CRpoint4 = []

    for i in xrange(int(len(pList))): # int(len(seed_points))
        p12,p23,p34,p41 = pList[i]
        SpineH = line_size(p41[0], p41[1], p12[0], p12[1])       # e1 Spine Height
        SpineW = line_size(p34[0], p34[1], p41[0], p41[1])       # e4 Spine Width
        H = max(SpineH,SpineW)
        W = min(SpineH,SpineW)
        ratio = W/H

        if (ratio <= 0.1):
            CRpoint4.append([p12,p23,p34,p41])

            # random color
            colorB = np.random.randint(256)
            colorG = np.random.randint(256)
            colorR = np.random.randint(256)
            color = (colorB,colorG,colorR)
            cv2.line(CRimg, (p12[0],p12[1]), (p23[0],p23[1]), color, 2, 8)
            cv2.line(CRimg, (p23[0],p23[1]), (p34[0],p34[1]), color, 2, 8)
            cv2.line(CRimg, (p34[0],p34[1]), (p41[0],p41[1]), color, 2, 8)
            cv2.line(CRimg, (p41[0],p41[1]), (p12[0],p12[1]), color, 2, 8)

    cv2.imwrite("CRimg.jpg",CRimg)
    #end for for-loop

    return CRpoint4
