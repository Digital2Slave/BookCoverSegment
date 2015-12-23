#!/usr/bin/python
# -*- encoding:UTF-8 -*-

import cv2
import numpy as np


def SelectImgFromCamera(aviFile=None):

    if (aviFile==None):
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(aviFile)
    Height, Width = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    ret0, prev = cam.read()
    prevgray  = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    chooseImg = np.ones((Height,Width,3), np.uint8)                             # select img from camera

    Th = 20                                                                     # select threshold value
    loop = ret0
    # keep looping over the frames
    while loop:
        # grab the current frame
        ret, img = cam.read()

        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #!<--------------------diffFrame processing------------------------
            diffFrame = cv2.absdiff(prevgray,gray)
            diffFlip = cv2.flip(diffFrame,1)                                    # mirror inversion between left and right

            LsdlineDetector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD) # create line detector
            Lsdlines, width, prec, nfa = LsdlineDetector.detect(diffFlip)       # gray image.

            if Lsdlines != None:
                a,b,c = Lsdlines.shape                                           # number of lines in diffFlip
                #!< draw lines on diffFlip
                for i in xrange(a):
                    pt1x = Lsdlines[i][0][0]
                    pt1y = Lsdlines[i][0][1]
                    pt2x = Lsdlines[i][0][2]
                    pt2y = Lsdlines[i][0][3]
                    #dist = np.sqrt(abs(pt1x-pt2x)**2+abs(pt1y-pt2y)**2)
                    #if (dist>=0.1*Height):
                    cv2.line(diffFlip, (pt1x, pt1y), (pt2x, pt2y), (255,255,255), 1, 8)
                cv2.imshow("LSDline", diffFlip)
                cv2.waitKey(10)
                #!< select choose frame.
                if (a>=0 and a<=Th):
                    chooseImg = img.copy()
                    ret = False
                    loop = False
            else:
                chooseImg = img.copy()
                ret = False
                loop = False
        else:
            loop = False
        #!< update
        prevgray = gray.copy()
    #!< end for While loop

    # cleanup the camera and close any open windows
    cam.release()
    cv2.destroyAllWindows()
    return chooseImg

def test():
    res = SelectImgFromCamera(aviFile=None)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':

    test()
