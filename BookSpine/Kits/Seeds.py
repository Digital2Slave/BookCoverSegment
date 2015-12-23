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

def zpoint(pij, v, d):
    '''
    zpoint: Set two points on the two sides of line's middle location.
    '''
    out_point = [0,0]
    out_point[0] = pij[0] + v[0]*d
    out_point[1] = pij[1] + v[1]*d
    return out_point

def DrawPointWithColor(img, points, color):
    for i in xrange(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        cv2.circle(img, (x,y), 1, color, -1, 8)
    #cv2.imshow("points", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite("points.jpg", img)

def DetectSeeds(Img,oolines):
    # vs and seed_points
    vs = []                                             # point2f direction of seed_points
    seed_points = []                                    # point seed_points

    for i in xrange(len(oolines)):

        pt1x = oolines[i][0]                            #第i条线第1个点pt1的x坐标值pt1x
        pt1y = oolines[i][1]                            #第i条线第1个点pt1的y坐标值pt1y
        pt2x = oolines[i][2]                            #第i条线第2个点pt2的x坐标值pt2x
        pt2y = oolines[i][3]                            #第i条线第2个点pt2的y坐标值pt2y

        linesize = line_size(pt1x, pt1y, pt2x, pt2y)
        dx = (pt2x-pt1x)/linesize
        dy = (pt2y-pt1y)/linesize
        v = [dx,dy]
        vt = [-dy,dx]
        fvt = [dy,-dx]

        startpoint = [pt1x, pt1y]
        linelongth = int(linesize)

        TwoSeeds = True
        if TwoSeeds:
            step  = linelongth/2
            zp    = zpoint(startpoint, v, step)
            seed1 = zpoint(zp, vt, 5)
            seed2 = zpoint(zp, fvt, 5)
            seed_points.append(seed1)
            seed_points.append(seed2)
            vs.append(v)
            vs.append(v)
        else:
            detaQP  = 10
            for j in xrange(0, linelongth, detaQP):
                zp = zpoint(startpoint, v, j)
                seed1 = zpoint(zp, vt, 10)
                seed2 = zpoint(zp, fvt, 10)
                seed_points.append(seed1)
                seed_points.append(seed2)
                vs.append(v)
                vs.append(v)
    #if len(seed_points) == 2*len(oolines):
    #    print("ok. each line in image owns two seeds(left and right on the middle of line!).")
    #DrawPointWithColor(Img,seed_points,(255,0,255))
    return seed_points,vs