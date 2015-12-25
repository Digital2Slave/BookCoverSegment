#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division    # Must be ahead of all
import os, argparse, cv2, copy
import numpy as np
import networkx as nx

from itertools import combinations
from Kits import LsdLine, Seeds, Point4, FilterCR,MWIS
from Kits import SelectImg

#!< Init of input image
def Init(img):
    '''
    InitOfVandS(Img):
    Input  : image
    Output : [sobelImg, angle, magnitude]
    '''
    # !< Denoising
    #img = cv2.blur(img, (5,5))

    # !< Extracting gradients on x and y directions
    sobelxf = cv2.Sobel(img, cv2.CV_32F, 3, 0, ksize = 5)
    sobelyf = cv2.Sobel(img, cv2.CV_32F, 0, 3, ksize = 5)

    # !< Computing angle and magnitude of img
    Size = np.shape(img)
    magnitude = np.zeros(Size, dtype = np.float32)
    angle = np.zeros(Size, dtype = np.float32)
    cv2.cartToPolar(sobelxf, sobelyf, magnitude, angle, False)  # True for angle in degree

    # !< To obtain Sobel img
    gradient = np.sqrt(cv2.convertScaleAbs(sobelxf)**2 + cv2.convertScaleAbs(sobelyf)**2)

    return gradient, angle

def DefValFlow(val, wall):
    '''
    DefValFlow
    :param val: x or y coordinate value
    :param wall: rows or cols of image
    :return:
    '''
    if (val < 0):
        val = 0

    while (val >= wall):
        val =  val - 1

    return val


#!< Draw Graph and Img
def DrawGraph(G,SavePicName):
    # positions for all nodes
    pos = nx.spring_layout(G)

    # nodes
    nx.draw_networkx_nodes(G,pos,node_size=200)

    # edges
    nx.draw_networkx_edges(G,pos,width=1.0)

    # labels
    nx.draw_networkx_labels(G,pos,font_size=8)

    plt.axis('off')
    plt.title(SavePicName)
    plt.savefig(SavePicName)
    plt.show()

def DrawLineOnImg(Img, cr, color):
    '''
    DrawLineOnImg
    :param Img: Input and Output Img
    :param cr: Candiate Rectangle
    :param color: tuple of three value
    :return: None
    '''
    p12,p23,p34,p41 = cr
    cv2.line(Img, (p12[0],p12[1]), (p23[0],p23[1]), color, 2, 8)
    cv2.line(Img, (p23[0],p23[1]), (p34[0],p34[1]), color, 2, 8)
    cv2.line(Img, (p34[0],p34[1]), (p41[0],p41[1]), color, 2, 8)
    cv2.line(Img, (p41[0],p41[1]), (p12[0],p12[1]), color, 2, 8)

def DrawLineTextOnImg(Img, cr, color, node, putText):
    '''
    DrawLineOnImg
    :param Img: Input and Output Img
    :param cr: Candiate Rectangle
    :param color: tuple of three value
    :return: None
    '''

    p12,p23,p34,p41 = cr
    cv2.line(Img, (p12[0],p12[1]), (p23[0],p23[1]), color, 2, 8)
    cv2.line(Img, (p23[0],p23[1]), (p34[0],p34[1]), color, 2, 8)
    cv2.line(Img, (p34[0],p34[1]), (p41[0],p41[1]), color, 2, 8)
    cv2.line(Img, (p41[0],p41[1]), (p12[0],p12[1]), color, 2, 8)

    if putText:
        rows,cols = np.shape(Img)[0:2]
        x = int((p23[0]+p41[0])/2)-5
        y = int((p23[1]+p41[1])/2)
        x = DefValFlow(x,cols)
        y = DefValFlow(y,rows)
        cv2.putText(Img,str(node),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,color)

def WriteSegmentBookSpine(BookSpineResult):
    '''
    Write Segmentated Book spine results to fixed directory.
    :param BookSpineResult:
    :return:
    '''
    if (len(BookSpineResult) == 0):
        print("Error! Not found one book spine.")
    else:
        for k,book in enumerate(BookSpineResult):
            index = './Example/Dst/'+str(k) + '.jpg'
            cv2.imwrite(index,book)


#!< Compute functions
def ComputeLine(p1,p2):
    '''
    Compute the line distance between two points
    :param p1: start point
    :param p2: end point
    :return:the distance between p1 and p2
    '''
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def ComputeNodeWeight(P4,v):
    '''
    ComputeNodeWeight
    :param P4: Set of CRs
    :param v: index of CR
    :return: node's weight value, that's weightV
    '''

    p12,p23,p34,p41 = P4[v]
    hv = ComputeLine(p12,p41) if (ComputeLine(p12,p41)>ComputeLine(p34,p41)) else ComputeLine(p34,p41)
    Ee1 = Point4.power(angle,p41,p12)
    Ee3 = Point4.power(angle,p34,p41)
    weight = hv*(1-(Ee1+Ee3)/(2*20))              # 2*Tg Tg = 20
    weightV = weight if (weight>0) else (-weight)
    return weightV

def CommonAreaRatio(rows, cols, cri, crj):
    '''
    :param cri:p12,p23,p34,p41 = cri
    :param crj:like to cri
    :return:ratio, ratio1 and ratio2
    '''
    H1 = max(ComputeLine(cri[0],cri[3]), ComputeLine(cri[2],cri[3]))
    W1 = min(ComputeLine(cri[0],cri[3]), ComputeLine(cri[2],cri[3]))
    ret1Aera = H1*W1

    H2 = max(ComputeLine(crj[0],crj[3]), ComputeLine(crj[2],crj[3]))
    W2 = min(ComputeLine(crj[0],crj[3]), ComputeLine(crj[2],crj[3]))
    ret2Aera = H2*W2

    img1 = np.zeros((rows,cols),np.uint8)
    img2 = copy.copy(img1)
    Mask1 = np.zeros((rows+2,cols+2),np.uint8)
    Mask2 = copy.copy(Mask1)

    seed1 = [int((cri[0][0]+cri[2][0])/2), int((cri[0][1]+cri[2][1])/2)]
    seed2 = [int((crj[0][0]+crj[2][0])/2), int((crj[0][1]+crj[2][1])/2)]
    DrawLineOnImg(img1,cri,(125))
    DrawLineOnImg(img2,crj,(125))

    retval1, img1, Mask1, rect1 = cv2.floodFill(img1,Mask1,(seed1[0],seed1[1]),125)
    retval2, img2, Mask2, rect2 = cv2.floodFill(img2,Mask2,(seed2[0],seed2[1]),125)

    img = img1 + img2
    retval, binaryImg = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)

    binaryImg, contours, hierarchy = cv2.findContours(binaryImg,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)

    Area = 0.0
    Len = len(contours)
    #print("len = %d" % Len)
    if (Len == 0):
        Area = 0.0
    elif(Len == 1):
        cnt = contours[0]
        Area = cv2.contourArea(cnt)
    else:
        AreaList = []
        for i in xrange(Len):
            icnt = contours[i]
            iArea = cv2.contourArea(icnt)
            AreaList.append(iArea)
        Area = max(AreaList)
    #print("Len = %d" % Area)

    ratio,ratioi, ratioj = 0.0, 0.0, 0.0
    if ((ret1Aera != 0) and (ret2Aera != 0)):
        ratioi = Area/ret1Aera
        ratioj = Area/ret2Aera
        ratio = Area/min(ret1Aera,ret2Aera)

    return ratio,ratioi,ratioj

def ExtractCRBottomLeftAndRight(P4,v):
    '''
    ExtractCRBottomLeftAndRight
    :param P4: Set of CR
    :param v: node index for CR
    :return: x coordinates of bottom left and bottom right
    '''
    Y =  [P4[v][0][1],P4[v][1][1],P4[v][2][1],P4[v][3][1]]
    ymax1 = max(Y)
    Y.remove(ymax1)
    ymax2 = max(Y)
    x1,x2 = 0,0
    for p in P4[v]:
        if p[1] == ymax1:
            x1 = p[0]
        elif p[1] == ymax2:
            x2 = p[0]
        else:
            continue
    xlv = min(x1,x2)         # like p41[0]
    xrv = max(x1,x2)         # like p34[0]
    return xlv,xrv


#!< Main Entry
if __name__ == "__main__":
    '''
    input : argv[1] or image path
    output: book spine segment images which store in './Example/Dst'
    Usage :
    $ python <name>.py --image=<imagepath>
    eg:
    $ python main.py --image='./Example/Src/002.jpg'
    '''

    '''
    # Load image for command line operator
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())
    InputImg = cv2.imread(args["image"]) # "./Example/Src/002.jpg"
    InputImg = cv2.imread('./Example/Src/008.jpg')
    '''
    cameraOpen = False
    Scale = 1.0
    if cameraOpen:
        Input = None     # like 'bookspine.avi'
        if (Input==None):
            InputImg = SelectImg.SelectImgFromCamera()
        else:
            InputImg = Input
            Scale = 4.0
    else:
        InputImg = cv2.imread('./Example/Src/003.jpg')
        Scale = 2.0

    Height, Width = np.shape(InputImg)[0:2]
    H = int(Height/Scale)
    W = int(Width/Scale)
    Img = cv2.resize(InputImg, (H,W),interpolation=cv2.INTER_NEAREST)
    GrayImg = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    rows, cols = np.shape(Img)[0:2]

    #-------------------------------------------------------------------------------------------------------------------

    #!< Sobel detection
    gradient, angle = Init(GrayImg)

    #!< Detected lines with OpenCV3.0 "2012 LSD algorithm" Done!
    oolines = LsdLine.LSD(Img)

    #!< Seeds_points and vs(dirction for each seeds_point) Done!
    seeds_points, vs = Seeds.DetectSeeds(Img,oolines)

    #!< To obtain pList
    pList = Point4.Detect4points(Img,angle,seeds_points,vs) # Core Function!!!
    #-------------------------------------------------------------------------------------------------------------------

    #!< Gain the CRs (candiate rectangle) whose ratio (eg.W/H of a single book spine) is smaller than 0.1
    CRimg = copy.copy(Img)
    CRpoint4 = FilterCR.FilterCRS(CRimg, pList) # Done!

    #-------------------------------------------------------------------------------------------------------------------
    #!< Create the graph G = (V,E)
    G = nx.Graph()    # undirected graph
    P4 = {}           # key:0~len(CRpoint4)  val:like [p12,p23,p34,p41]
    ratioV = 0.1      # common area ratio between cri and crj
    for vi,cri in enumerate(CRpoint4):
        G.add_node(vi)
        P4[vi] = cri
        # compute the edge (vi,vj) which belongs to E
        for vj, crj in enumerate(CRpoint4):
            if (vj != vi):
                ratio,ratio1,ratio2 = CommonAreaRatio(rows, cols, cri, crj)
                if (ratio >= ratioV):
                    G.add_edge(vi,vj)
                    #G.add_weighted_edges_from([(vi,vj,ratio)])
    #DrawGraph(G,"Graph.jpg")

    #-------------------------------------------------------------------------------------------------------------------

    #!< Gain weight for each node in G
    N = G.nodes()                                               # nodes of G
    E = G.edges()                                               # edges of G
    Appequal = 30                                               # x Coordinate offset

    for v in N:                                                 # for loop for each node in G
        HSv = G.neighbors(v)                                    # v's neighbors nodes in G
        HSvLen = len(HSv)
        Condition1 = True if HSvLen>= 2 else False
        Condition2, Condition3 = False, False
        if Condition1:                                          # Condition1 hold!!!
            for u1,u2 in combinations(HSv,2):
                if (u1,u2) not in E:
                    Condition2 = True                           # be sure for any pair of (u1,u2) in HSv not belong to E
                else:
                    Condition2 = False
                    break
            if Condition2:                                      # Condition2 hold!!!
                #print("Condition2 node = %d" % v)
                #G.node[v] = 0
                #break
                xlv,xrv = ExtractCRBottomLeftAndRight(P4,v)     # obtain x coordinates of bottom left and bottom right
                xL, xR, xLR = False, False, False               # be sure bottom left and bottom right
                upie = 0                                        # used to store u'
                for u,t in combinations(HSv,2):                 # u,t corresponding to u u'
                    xLu,xRu = ExtractCRBottomLeftAndRight(P4,u) # xLu corresponding to bottom left of u
                    tLu,tRu = ExtractCRBottomLeftAndRight(P4,t) # tRu corresponding to bottom right of u'
                    if (abs(xLu-xlv)<=Appequal):                # xLu ≈ xlv exist u ∈ HSv
                        xL = True
                    if (abs(tRu-xrv)<=Appequal):                # tRu ≈ xrv exist u' ∈ HSV
                        xR = True
                        upie = t                                # for next operate ui != xRuev, that is ui != u'
                    if (xL and xR):
                        xLR = True                              # exist u,u'∈ HS(v) and xLu ≈ xlv, tRu ≈ xrv
                        break
                if xLR:
                    for ui in HSv:                                           # for each ui ∈ HSv
                        xLi, xRi = ExtractCRBottomLeftAndRight(P4,ui)        # xRi
                        if ui != upie:                                       # ui ≠ upie, that is ui ≠ u'
                            for uj in HSv:                                   # exist uj ∈ HSv
                                xLj,xRj = ExtractCRBottomLeftAndRight(P4,uj) # xLj
                                if (abs(xRi-xLj)<=Appequal):                 # xRi ≈ xLj
                                    Condition3 = True
                                    break
                        else:
                            continue
                if Condition3:                                               # Condition3 hold!!!
                    '''
                    Another common spatial relation between PRs is a single PR that covers a set of PRs
                    that corresponds to adjacent book spines.check whether HS(v) belongs to N(v) or not.
                    G.node[v] = weight
                    '''
                    G.node[v] = 0                                            # weight = 0, discard current v
                #  condition3
            # condition2
        # condition1
        if((Condition1 == False) or (Condition2 == False) or (Condition3 == False)):
            '''
            The most common spatial relation between PRs is due to multiple detections of the same spine
            but truncated at different heights.G.node[v] = weight
            '''
            G.node[v] = ComputeNodeWeight(P4,v)
    #for each node in G

    #-------------------------------------------------------------------------------------------------------------------

    '''
    #mis = [4,6,14,36,41,43,45,46,52,54,59,63,74] # manual selected for ./Example/002.jpg
    '''
    n = len(N)
    A = np.zeros((n**2),np.uint8)      # default np.uint8, N = G.nodes()
    for i in xrange(n):
        for j in xrange(i+1,n):
            if (i,j) in E:
                A[n*i+j] = A[n*j+i] = 1
    # end for A

    W = np.zeros((n),np.float16)       # weight of G's node
    for i in N:
        if (G.node[i] == 0):
            W[i] = 0
        else:
            W[i] = G.node[i]
    # end for W

    X = np.zeros((n),np.uint8)
    X = MWIS.mwisGetX(W,A,n,X)         # MWIS based on integer fixed point
    Wmis = [ index for index,val in enumerate(X) if (val==1)]

    #-------------------------------------------------------------------------------------------------------------------

    Outimg = copy.copy(Img)          # write output image
    BookSpineResult = []             # store the results of book spine in a bookspine image
    IndexNode = Wmis                 # store the final node index of Candiate Rectangle (CR) for a bookspine image
    for node in IndexNode:
        p12,p23,p34,p41 = P4[node]
        seed = [int((p12[0]+p34[0])/2), int((p12[1]+p34[1])/2)]
        BookSpine = np.zeros((rows,cols),np.uint8)
        BookMask = np.zeros((rows+2,cols+2),np.uint8)
        color = (255)
        cv2.line(BookSpine, (p12[0],p12[1]), (p23[0],p23[1]), color, 2, 8)
        cv2.line(BookSpine, (p23[0],p23[1]), (p34[0],p34[1]), color, 2, 8)
        cv2.line(BookSpine, (p34[0],p34[1]), (p41[0],p41[1]), color, 2, 8)
        cv2.line(BookSpine, (p41[0],p41[1]), (p12[0],p12[1]), color, 2, 8)
        ret, BookSpine, BookMask, rect = cv2.floodFill(BookSpine,BookMask,(seed[0],seed[1]),255)

        # chip of book spine result for writing to fixed directory
        xmin = min(p12[0],p23[0],p34[0],p41[0])
        xmax = max(p12[0],p23[0],p34[0],p41[0])
        ymin = min(p12[1],p23[1],p34[1],p41[1])
        ymax = max(p12[1],p23[1],p34[1],p41[1])
        row = ymax - ymin
        col = xmax - xmin

        BookRect = BookSpine[ymin:ymax,xmin:xmax]
        BookRect = cv2.cvtColor(BookRect,cv2.COLOR_GRAY2BGR)
        ImgRect  = Img[ymin:ymax,xmin:xmax]
        result   = cv2.bitwise_and(BookRect,ImgRect)
        BookSpineResult.append(result)

        # Draw final selected rectangle to Outimg
        DrawLineTextOnImg(Outimg,P4[node], (0,0,255), node, False) # True for puttext in rectangle
    # end for loop in IndexNode
    OutputImageFlip = cv2.flip(Outimg,1)
    cv2.imwrite("OutputImage.jpg",OutputImageFlip)
    cv2.imshow("output", OutputImageFlip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #-------------------------------------------------------------------------------------------------------------------

    #!< write the results of book spine to current dir
    WriteSegmentBookSpine(BookSpineResult)

    #-------------------------------------------------------------------------------------------------------------------
