#-*- coding: UTF-8 -*-

import cv2
import numpy as np

def LSD(Img):

    rows,cols,channels = np.shape(Img)
    if (channels==3):
        GrayImg = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    else:
        GrayImg = Img

    # !< LSD line detection http://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#linesegmentdetector
    LineDescriptor = cv2.createLineSegmentDetector(
        _refine=cv2.LSD_REFINE_STD, #E.g. breaking arches into smaller straighter line approximations. cv2.LSD_REFINE_STD
        _scale= 0.8,                #the scale of the image that will be used to find the lines. Range (0..1]. 0.8
        _sigma_scale=0.6,           #Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale. 0.6
        _quant=2.0,                 #Bound to the quantization error on the gradient norm. 2.0
        _ang_th=22.5,               #Gradient angle tolerance in degrees. 22.5
        _log_eps=0,                 #Detection threshold: -log10(NFA) > log_eps. Used only when advancent refinement is chosen. 0
        _density_th=0.7,            # Minimal density of aligned region points in the enclosing rectangle. 0.7
        _n_bins=1024)               #Number of bins in pseudo-ordering of gradient modulus. 1024

    #LineDescriptor = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    LDlines, width, prec, nfa = LineDescriptor.detect(GrayImg)
    LDlines = np.int0(LDlines)

    oolines = []   # set of output lines

    #!< Starting by yince's algorithm
    #mask = np.zeros((rows,cols),np.uint8)
    a,b,c = LDlines.shape

    for i in xrange(a):
        pt1x = LDlines[i][0][0]
        pt1y = LDlines[i][0][1]
        pt2x = LDlines[i][0][2]
        pt2y = LDlines[i][0][3]
        dist = np.sqrt(abs(pt1x-pt2x)**2+abs(pt1y-pt2y)**2)

        if (dist>=0.1*rows):
            oolines.append([pt1x,pt1y,pt2x,pt2y])
            #cv2.line(mask, (pt1x, pt1y), (pt2x, pt2y), (255,255,255), 1, 8)
    #cv2.imshow("LSDline", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return oolines