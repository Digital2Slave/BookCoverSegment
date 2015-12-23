# -*- coding:UTF-8 -*-

from __future__ import division
import numpy as np
from numpy import fabs
import copy

alpha = 2

def mwisComputeObjFuncX(w, A, n, x):

    goal, val = 0.0, 0.0

    for i in xrange(n):
        for j in xrange(n):
            val += A[i*n+j]*x[i]*x[j]

    goal = - 0.5*alpha*val

    for i in xrange(n):
        goal += w[i]*x[i]

    return  goal

def mwisComputeObjFuncY(w, A, n, y):

    goal, val = 0.0, 0.0

    for i in xrange(n):
        for j in xrange(n):
            val += A[i*n+j]*y[i]*y[j]

    goal = - 0.5*alpha*val

    for i in xrange(n):
        goal += w[i]*y[i]

    return goal

def mwisConstraint(A, n, y):
    val = 0.0

    for i in xrange(n):
        for j in xrange(n):
            val += A[i*n+j]*y[i]*y[j]
    return val

def mwisGetX(w, A, n, x):
    """
    MWIS based on integer fixed point
    """
    A = np.float16(A)
    maxIter = 1000
    maxw, minw, norm2, threshold, rangew = -1e20, 1e20, 1e20, 1e-16, 0.0
    eta, c, d, val, maxval = 0.0, 0.0, 0.0, 0.0, 0.0
    maxi = 0            # int
    flag = np.uint8(0)  # unsigned char
    objf, objftemp = 0.0, 0.0

    z = np.zeros((n), np.float16)
    y = np.zeros((n), np.float16)
    xtilde = np.zeros((n), np.uint8)

    if (x == None) or (len(x) == 0):
        x = np.zeros((n), np.uint8)

    # init y, x, and rescale w
    for i in xrange(n):
        if (maxw < w[i]):
            maxw = w[i]
            j = i
        if (minw > w[i]):
            minw = w[i]
    rangew = maxw - minw

    for i in xrange(n):
        w[i] = (w[i]-minw)/rangew
        y[i] = (0.5*w[i]+0.01)

    # compute objective function
    objf = mwisComputeObjFuncY(w, A, n, y)
    #print("objf: %f\n" % objf)

    #iterate
    while(maxIter and (norm2 > threshold)):
        maxval = -1e20
        maxi = -1
        flag = np.uint8(0)
        #find gradient dy
        for i in xrange(n):
            val = 0
            for j in xrange(n):
                val += A[i*n+j]*y[j]
            val = w[i] - alpha*val
            xtilde[i] = 1 if (val>=0) else 0           # very important!!!
            flag |= xtilde[i]
            if (maxval<val):
                maxval = val
                maxi = i
        if (not flag):
            xtilde[maxi] = 1

        #compute objective function with xtilde
        objftemp = mwisComputeObjFuncY(w, A, n, xtilde)

        #check if objective function increase
        if (objftemp>=objf):
            #compute norm
            norm2 = 0.0
            for i in xrange(n):
                norm2 += (y[i]-xtilde[i])**2
            #update y and x
            for i in xrange(n):
                y[i] = xtilde[i]
            x = copy.copy(xtilde)
            objf = objftemp
        else:
            #compute z = xtilde - y
            for i in xrange(n):
                z[i] = xtilde[i] - y[i]

            #compute eta
            c = d = 0.0

            #(w-alpha*Ay)z
            for i in xrange(n):
                val = 0
                for j in xrange(n):
                    val += A[i*n+j]*y[j]
                c += (w[i]-alpha*val)*z[i]

            #zAz
            for i in xrange(n):
                for j in xrange(n):
                    d += A[i*n+j]*z[i]*z[j]

            #compute eta
            #divisionVal = (c/(alpha*d)) if (d!=0) else (c/alpha)
            divisionVal = c/(alpha*d)
            eta = max(divisionVal,0) if (d>0) else min(divisionVal,1)

            #compute norm
            if (eta == 0):
                norm2 = 0
                y.fill(0)
            else:
                norm2 = 0
                for i in xrange(n):
                    norm2 += (eta*z[i])**2
                #compute y
                for i in xrange(n):
                    y[i] += eta*z[i]

            #compute objective function with y
            objf = mwisComputeObjFuncY(w, A, n ,y)
        #end for else
        maxIter -= 1
    #end for iterate
    #print("objf: %f\n" % objf)

    return  x