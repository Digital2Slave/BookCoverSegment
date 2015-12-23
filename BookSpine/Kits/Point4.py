#-*- coding: UTF-8 -*- 

import cv2
import numpy as np

def DefValFlow(val, wall):

    if (val < 0):
        val = 0
    while (val >= wall):
        val =  val - 1

    return val

def power(angle, a, b):
    '''
    power: energy function for edge
    '''
    rows,cols = np.shape(angle)[0:2]

    LenOfab = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    dx = (b[0]-a[0])/LenOfab
    dy = (b[1]-a[1])/LenOfab

    sum = 0.0
    n = 0
    for i in xrange(0, int(LenOfab), 5):
        y = int(a[1] + i*dy)
        x = int(a[0] + i*dx)
        # 防止 y and x 值溢出
        y = DefValFlow(y, rows)
        x = DefValFlow(x, cols)

        angleV = angle[y, x]                # in radian
        #if (angleV < (np.pi*2+1)) and (angleV > -(np.pi/180)):
        cosV = np.cos(angleV)
        sinV = np.sin(angleV)
        #absV = abs(dx*cosV + dy*sinV)
        absV = np.sqrt((dx*cosV)**2+(dy*sinV)**2)
        absV = 1 if (absV>1) else (-1 if (absV<-1) else absV)
        sum += np.arccos(absV)
        n +=1
    if (n==0):
        return 90
    else:
        return np.degrees(sum/float(n))

def zpoint(pij, v, d):
    '''
    zpoint: Set two points on the two sides of line's points.
    '''
    out_point = [0,0]
    out_point[0] = pij[0] + v[0]*d
    out_point[1] = pij[1] + v[1]*d
    return out_point

def prpoint(center, di, dj, vi, vj):
    '''
    prpoint: p12,p23,p34,p41
    '''
    
    Denoi = np.sqrt(vi[0]**2+vi[1]**2)   # i分母:Denominator
    Denoj = np.sqrt(vj[0]**2+vj[1]**2)   # j分母:Denominator
    
    # qi and qj
    qi = [0,0]
    qi[0] = center[0] + di*(vi[0]/Denoi) # (qi-center)/di = vi/Denoi
    qi[1] = center[1] + di*(vi[1]/Denoi)
    
    qj = [0,0]
    qj[0] = center[0] + dj*(vj[0]/Denoj) # (qj-center)/dj = vj/Denoj
    qj[1] = center[1] + dj*(vj[1]/Denoj)
    
    # p1 and p2
    p1 = [0,0]
    p1[0] = qi[0]+dj*(vj[0]/Denoj)       # (p1-qi)/dj = vj/Denoj
    p1[1] = qi[1]+dj*(vj[1]/Denoj)
    
    p2 = [0,0]
    p2[0] = qj[0]+di*(vi[0]/Denoi)       # (p2-qj)/di = vi/Denoi
    p2[1] = qj[1]+di*(vi[1]/Denoi)

    # pij
    pij   = [0,0]                        # pij like to p12,p23,p34,p41
    pij[0] = int((p1[0]+p2[0])/2)
    pij[1] = int((p1[1]+p2[1])/2)
    return pij

def BoolValOfLocationP1ToP4(p,cols,rows):
    '''
    define bool value of location_p1 to p4
    '''
    if (p[0]<cols and p[0]>0) and (p[1]<rows and p[1]>0):
        return True
    else:
        return False


def DiAndActivedi(angle, p1, p2, v, zd, di, activedi, e, ismine, Tg, BetaTg, LumTg):

    if (activedi == True):
        zp1 = zpoint(p1, v, zd)
        zp2 = zpoint(p2, v, zd)
        detaEe1 =  power(angle, p1, zp1)
        detaEe2 =  power(angle, p2, zp2)
        maxOfdeta = max(detaEe1,detaEe2)
        if (abs(e-90)<Tg) and (ismine>abs(e-90)) and (abs(maxOfdeta-90)>BetaTg):
            activedi = False
            ismine = abs(e-90)
        else:
            di += 1
    else:
        if (abs(e-90)>LumTg):
            activedi = True
    return di,activedi,ismine


def Detect4points(Img, angle, seed_points, vs):
    '''
    Select 4 points
    :param Img:
    :param angle:
    :param seed_points:
    :param vs:
    :return:
    '''

    rows,cols = np.shape(Img)[0:2]
    Tg, BetaTg, LumTg = 20.0, 30.0, 40.0 # Tg BetaTg LumTg 20,30,40
    pList = []

    for i in xrange(int(len(seed_points))): # int(len(seed_points))
        dx,dy = vs[i]
        v     = [dx,dy]
        vt    = [-dy,dx]
        fvt   = [dy, -dx]
        fv    = [-dx,-dy]

        p12 = [0, 0]
        p23 = [0, 0]
        p34 = [0, 0]
        p41 = [0, 0]
        d1,          d2,          d3,          d4          = 5, 5, 5, 5
        active_d1,   active_d2,   active_d3,   active_d4   = True, True, True, True
        location_p1, location_p2, location_p3, location_p4 = True, True, True, True
        iterations= True

        while ((active_d1 or active_d2 or active_d3 or active_d4) and iterations):

            p12 = prpoint(seed_points[i], d1, d2, fvt, fv)
            p23 = prpoint(seed_points[i], d2, d3, fv, vt)
            p34 = prpoint(seed_points[i], d3, d4, vt, v)
            p41 = prpoint(seed_points[i], d4, d1, v, fvt)

            location_p1 = BoolValOfLocationP1ToP4(p12, cols, rows)
            location_p2 = BoolValOfLocationP1ToP4(p23, cols, rows)
            location_p3 = BoolValOfLocationP1ToP4(p34, cols, rows)
            location_p4 = BoolValOfLocationP1ToP4(p41, cols, rows)

            if (location_p1 and location_p2 and location_p3 and location_p4):
                iterations = True
            else:
                iterations = False

            e1 = power(angle, p12, p23)
            e2 = power(angle, p23, p34)
            e3 = power(angle, p34, p41)
            e4 = power(angle, p41, p12)

            # coding
            zd = 5   #pi/36
            ismine = min(e1,e2,e3,e4)
            e_min1, e_min2, e_min3, e_min4 = ismine, ismine, ismine, ismine

            d2,active_d1,e_min1 = DiAndActivedi(angle, p12, p23, fv,  zd, d2, active_d1, e1, e_min1, Tg, BetaTg, LumTg)
            d3,active_d2,e_min2 = DiAndActivedi(angle, p23, p34, vt,  zd, d3, active_d2, e2, e_min2, Tg, BetaTg, LumTg)
            d4,active_d3,e_min3 = DiAndActivedi(angle, p34, p41, v,   zd, d4, active_d3, e3, e_min3, Tg, BetaTg, LumTg)
            d1,active_d4,e_min4 = DiAndActivedi(angle, p41, p12, fvt, zd, d1, active_d4, e4, e_min4, Tg, BetaTg, LumTg)

        # end for while loop
        pList.append([p12,p23,p34,p41])

        '''
        color = (0,255,255)
        cv2.line(Img, (p12[0],p12[1]), (p23[0],p23[1]), color, 2, 8)
        cv2.line(Img, (p23[0],p23[1]), (p34[0],p34[1]), color, 2, 8)
        cv2.line(Img, (p34[0],p34[1]), (p41[0],p41[1]), color, 2, 8)
        cv2.line(Img, (p41[0],p41[1]), (p12[0],p12[1]), color, 2, 8)
        cv2.imshow("CR",Img)
        cv2.waitKey(5)
        '''

    # end for seed_points loop
    return pList
