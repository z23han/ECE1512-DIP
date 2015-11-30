import os
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy as sci
import scipy.misc as misc
import sys
import cv2


'''
Module for Canny edge detection
Requirements: 1.scipy.(numpy is also mandatory, but it is assumed to be
                      installed with scipy)
              2. Python Image Library (only for viewing the final image.)
Author: Vishwanath
contact: vishwa.hyd@gmail.com
'''
try:
    import Image
except ImportError:
    print 'PIL not found. You cannot view the image'


from scipy import *
from scipy.ndimage import *
from scipy.signal import convolve2d as conv


def colorImSave(filename, array):
    imArray = misc.imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        misc.imsave(filename, cm.jet(imArray))
    else:
        misc.imsave(filename, imArray)


def canny(im, sigma, thresHigh = 50, thresLow = 10, version='edge', kernel='sobel'):
    '''
        Takes an input image in the range [0, 1] and generate a gradient image
        with edges marked by 1 pixels.
    '''
    imin = im.copy() * 255.0

    # Create the gauss kernel for blurring the input image
    # It will be convolved with the image
    # wsize should be an odd number
    wsize = 5
    gausskernel = gaussFilter(sigma, window = wsize)
    # fx is the filter for vertical gradient
    # fy is the filter for horizontal gradient
    # Please not the vertical direction is positive X

    if kernel == 'prewitt':
        fx = createFilter([-1, 0, 1,
                           -1, 0, 1,
                           -1, 0, 1])
        fy = createFilter([-1, -1, -1,
                           0, 0, 0,
                           1, 1, 1])
    elif kernel == 'sobel':
        fx = createFilter([-1, 0, 1,
                           -2, 0, 2,
                           -1, 0, 1])
        fy = createFilter([-1, -2, -1,
                           0, 0, 0,
                           1, 2, 1])
    elif kernel == 'laplacian':
        fx = createFilter([0, -1, 0,
                           -1, 4, 1,
                           0, -1, 0])
        fy = createFilter([0, -1, 0,
                           -1, 4, 1,
                           0, -1, 0])
    '''
    fx = createFilter([0, 1, 0,
                       0, 0, 0,
                       0, -1, 0])

    fy = createFilter([0, 0, 0,
                       -1, 0, 1,
                       0, 0, 0])
    '''

    imout = conv(imin, gausskernel, 'valid')
    gradxx = conv(imout, fx, 'valid')
    gradyy = conv(imout, fy, 'valid')

    gradx = np.zeros(im.shape)
    grady = np.zeros(im.shape)
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy

    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    grad = hypot(gradx, grady)
    theta = arctan2(grady, gradx)
    theta = 180 + (180 / pi) * theta
    # Only significant magnitudes are considered. All others are removed
    xx, yy = where(grad < 10)
    theta[xx, yy] = 0
    grad[xx, yy] = 0

    # The angles are quantized. This is the first step in non-maximum
    # suppression. Since, any pixel will have only 4 approach directions.
    x0,y0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)
                   +(theta>337.5)) == True)
    x45,y45 = where( ((theta>22.5)*(theta<67.5)
                      +(theta>202.5)*(theta<247.5)) == True)
    x90,y90 = where( ((theta>67.5)*(theta<112.5)
                      +(theta>247.5)*(theta<292.5)) == True)
    x135,y135 = where( ((theta>112.5)*(theta<157.5)
                        +(theta>292.5)*(theta<337.5)) == True)

    theta = theta

    # return theta without any modifications
    if version == 'theta2':
        return theta * np.pi / 180


    #Image.fromarray(theta).convert('L').save('Angle map.jpg')
    theta[x0,y0] = 0
    theta[x45,y45] = 45
    theta[x90,y90] = 90
    theta[x135,y135] = 135

    # put theta into [0, 45, 90, 136]
    if version == 'theta1':
        return theta * np.pi / 180

    x,y = theta.shape
    temp = Image.new('RGB',(y,x),(255,255,255))
    for i in range(x):
        for j in range(y):
            if theta[i,j] == 0:
                temp.putpixel((j,i),(0,0,255))
            elif theta[i,j] == 45:
                temp.putpixel((j,i),(255,0,0))
            elif theta[i,j] == 90:
                temp.putpixel((j,i),(255,255,0))
            elif theta[i,j] == 45:
                temp.putpixel((j,i),(0,255,0))
    retgrad = grad.copy()
    x,y = retgrad.shape

    for i in range(x):
        for j in range(y):
            if theta[i,j] == 0:
                test = nms_check(grad,i,j,1,0,-1,0)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 45:
                test = nms_check(grad,i,j,1,-1,-1,1)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 90:
                test = nms_check(grad,i,j,0,1,0,-1)
                if not test:
                    retgrad[i,j] = 0
            elif theta[i,j] == 135:
                test = nms_check(grad,i,j,1,1,-1,-1)
                if not test:
                    retgrad[i,j] = 0

    init_point = stop(retgrad, thresHigh)
    # Hysteresis tracking. Since we know that significant edges are
    # continuous contours, we will exploit the same.
    # thresHigh is used to track the starting point of edges and
    # thresLow is used to track the whole edge till end of the edge.

    while (init_point != -1):
        #Image.fromarray(retgrad).show()
        # print 'next segment at',init_point
        retgrad[init_point[0],init_point[1]] = -1
        p2 = init_point
        p1 = init_point
        p0 = init_point
        p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        while (p0 != -1):
            #print p0
            p2 = p1
            p1 = p0
            retgrad[p0[0],p0[1]] = -1
            p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        init_point = stop(retgrad,thresHigh)

    # Finally, convert the image into a binary image
    x,y = where(retgrad == -1)
    retgrad[:,:] = 0
    retgrad[x,y] = 1.0
    return retgrad


def createFilter(rawfilter):
    '''
        This method is used to create an NxN matrix to be used as a filter,
        given a N*N list
    '''
    order = pow(len(rawfilter), 0.5)
    order = int(order)
    filt_array = array(rawfilter)
    outfilter = filt_array.reshape((order,order))
    return outfilter


def gaussFilter(sigma, window = 3):
    '''
        This method is used to create a gaussian kernel to be used
        for the blurring purpose. inputs are sigma and the window size
    '''
    kernel = zeros((window,window))
    c0 = window // 2

    for x in range(window):
        for y in range(window):
            r = hypot((x-c0),(y-c0))
            val = (1.0/(2*pi*sigma*sigma))*exp(-(r*r)/(2*sigma*sigma))
            kernel[x,y] = val
    return kernel / kernel.sum()


def nms_check(grad, i, j, x1, y1, x2, y2):
    '''
        Method for non maximum suppression check. A gradient point is an
        edge only if the gradient magnitude and the slope agree

        for example, consider a horizontal edge. if the angle of gradient
        is 0 degrees, it is an edge point only if the value of gradient
        at that point is greater than its top and bottom neighbours.
    '''
    try:
        if (grad[i,j] > grad[i+x1,j+y1]) and (grad[i,j] > grad[i+x2,j+y2]):
            return 1
        else:
            return 0
    except IndexError:
        return -1


def stop(im, thres):
    '''
        This method is used to find the starting point of an edge.
    '''
    X,Y = where(im > thres)
    try:
        y = Y.min()
    except:
        return -1
    X = X.tolist()
    Y = Y.tolist()
    index = Y.index(y)
    x = X[index]
    return [x,y]


def nextNbd(im, p0, p1, p2, thres):
    '''
        This method is used to return the next point on the edge.
    '''
    kit = [-1,0,1]
    X,Y = im.shape
    for i in kit:
        for j in kit:
            if (i+j) == 0:
                continue
            x = p0[0]+i
            y = p0[1]+j

            if (x<0) or (y<0) or (x>=X) or (y>=Y):
                continue
            if ([x,y] == p1) or ([x,y] == p2):
                continue
            if (im[x,y] > thres): #and (im[i,j] < 256):
                return [x,y]
    return -1



def markStroke(mrkd, p0, p1, rad, val):
    # mark the pixels that will be painted by a stroke from pixel
    # p0=(x0,y0) to pixel p1=(x1,y1). These pixels are set to val
    # in the ny x nx double array mrkd. The paintbrush is circular
    # with radius rad>0

    sizeIm = mrkd.shape
    sizeIm = sizeIm[0:2]
    nx = sizeIm[1]
    ny = sizeIm[0]
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    rad = max(rad, 1)
    # bounding box
    # stack array vertically
    concat = np.vstack([p0, p1])
    # min along the first axis, which is vertical
    bb0 = np.floor(np.amin(concat, axis=0))-rad
    # max along the first axis, which is vertical
    bb1 = np.ceil(np.amax(concat, axis=0))+rad
    # check for intersection of bounding box with image
    intersect = 1
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        intersect = 0
    if intersect:
        # crop the bounding box
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1], 1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1], ny])]), axis=1)
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1], 1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1], ny])]), axis=1)
        # compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1
        szBB = [tmp[1], tmp[0]]
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        t = q1 - q0
        nrmt = np.linalg.norm(t)
        [x,y] = np.meshgrid(np.array([i+1 for i in range(int(szBB[1]))]), np.array([i+1 for i in range(int(szBB[0]))]))
        d = np.zeros(szBB)
        d.fill(float('inf'))

        if nrmt == 0:
            # use distance to point p0
            d = np.sqrt((x - q0[0])**2 + (y - q0[1])**2)
            idx = (d <= rad)
        else:
            # use distance to segment q0, q1
            t = t/nrmt
            n = [t[1], -t[0]]
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            idx = (tmp >= 0) & (tmp <= nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt((x[np.where(idx)] - q0[0])**2 + (y[np.where(idx)] - q0[1])**2)
            idx = (tmp > nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt((x[np.where(idx)] - q1[0])**2 + (y[np.where(idx)] - q1[1])**2)

            # pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        # mark the pixel
        if np.any(idx.flatten('F')):
            xy = (bb0[1]-1+y[np.where(idx)] + sizeIm[0] * (bb0[0]+x[np.where(idx)]-2)).astype(int)
            sz = mrkd.shape
            m = mrkd.flatten('F')
            m[xy-1] = val
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order='F')

    return mrkd


def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # paint a stroke from pixel p0=(x0, y0) to pixel p1=(x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # the stroke has rgb values given by colour (a 3x1 vector, with values
    # in [0,1]. the paintbrush is circular with radius rad>0)
    sizeIm = canvas.shape
    sizeIm = sizeIm[0:2]
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
    # paint
    # flatten is modify the array into 1d, and F is go down first
    if np.any(idx.flatten('F')):
        canvas = np.reshape(canvas, (np.prod(sizeIm), 3), 'F')
        xy = y[idx] + sizeIm[0] * (x[idx]-1)
        canvas[xy-1,:] = np.tile(np.transpose(colour[:]), (len(xy),1))
        canvas = np.reshape(canvas, sizeIm+(3,), 'F')
    return canvas


# Computing Canny Edges, using the function provided by Vishwanath
# Use the image intensity suggested by Litwinowicz I(x) = 0.3*R(x) + 0.59*G(x) + 0.11*B(x)
def cannyEdge(imRGB, sigma, thresHigh=50, thresLow=10, version='edge', kernel='sobel'):
    im = imRGB.copy()
    intIm = np.zeros(im.shape[0:2])
    pixR = 0.0
    pixG = 0.0
    pixB = 0.0
    pixInt = 0.0
    for i in range(len(im)):
        for j in range(len(im[i])):
            pixR = im[i][j][0]
            pixG = im[i][j][1]
            pixB = im[i][j][2]
            pixInt = 0.3*pixR + 0.59*pixG + 0.11*pixB
            intIm[i][j] = pixInt

    # return thetas within [0, 45, 90, 135]
    if version == 'theta1':
        thetas = canny(intIm, sigma, thresHigh, thresLow, version='theta1', kernel=kernel)
        return thetas
    # return thetas calculated
    elif version == 'theta2':
        thetas = canny(intIm, sigma, thresHigh, thresLow, version='theta2', kernel=kernel)
        return thetas

    retgrad = canny(intIm, sigma, thresHigh,thresLow)
    return retgrad


# we need to find the end point of one stroke
def findEndPoint(imRGB, cntr, delta, halfLen, sizeIm):
    # cntr is the center of the stroke, delta is the tangent direction, sizeIm is the size of the image
    # imRGB is the canny edge detection binary image
    if cntr[0] > sizeIm[1] or cntr[0] < 0 or cntr[1] > sizeIm[0] or cntr[1] < 0:
        print 'ERROR input for cntr'
        return

    if abs(delta[0]) > abs(delta[1]):
        delta = delta / abs(delta[0])
    else:
        delta = delta / abs(delta[1])

    k = 1
    endPoint1 = cntr
    endPoint2 = cntr
    end1Found = False
    end2Found = False
    while True:
        endPoint1 = cntr + np.round(k * delta)
        if norm(endPoint1 - cntr) > halfLen:
            break
        if endPoint1[0] > sizeIm[1] or endPoint1[1] >= sizeIm[0]:
            endPoint1 = endPoint1 - abs(np.round(delta))
            break
        if endPoint1[0] < 0 or endPoint1[1] < 0:
            endPoint1 = endPoint1 + abs(np.round(delta))
            break

        if imRGB[endPoint1[1]-1, endPoint1[0]-1] == 1:
            end1Found = True
        if end1Found:
            break
        k += 1

    k = 1
    while True:
        endPoint2 = cntr - np.round(k * delta)
        if norm(endPoint2 - cntr) > halfLen:
            break
        if endPoint2[0] > sizeIm[1] or endPoint2[1] >= sizeIm[0]:
            endPoint2 = endPoint2 - abs(np.round(delta))
            break
        if endPoint2[0] < 0 or endPoint2[1] < 0:
            endPoint2 = endPoint2 + abs(np.round(delta))
            break

        if imRGB[endPoint2[1]-1, endPoint2[0]-1] == 1:
            end2Found = True
        if end2Found:
            break
        k += 1

    return endPoint1, endPoint2


'''
input guidance:
    colorPlateNumber == 1 ::=> original image
    colorPlateNumber == 2 ::=> edge detection image
    colorPlateNumber == 3 ::=> fixed theta = 45
    colorPlateNumber == 4 ::=> random fixed theta
    colorPlateNumber == 5 ::=> thetas in [0, 45, 90, 135]
    colorPlateNumber == 6 ::=> calculated thetas from gradient edges
    colorPlateNumber == 7 ::=> random perturbation on thetas and intensities
'''

if __name__ == '__main__':

    try:
        imageFile = sys.argv[1]
    except:
        print 'python impressionist_painting.py imageFile colorPlateNumber'
        sys.exit()

    try:
        colorPlateNum = int(sys.argv[2])
    except:
        print 'python impressionist_painting.py imageFile colorPlateNumber'
        sys.exit()

    if colorPlateNum not in range(1,9):
        print 'ERROR in color plate number'
        sys.exit()

    # read image and convert it to double, and scale R,G,B
    # channel to range [0,1]
    imRGB = array(Image.open(imageFile))
    imRGB = np.double(imRGB)/255.0

    plt.clf()
    plt.axis('off')

    ################# colorPlateNumber = 1 ###################
    if colorPlateNum == 1:
        plt.imshow(imRGB)
        plt.pause(3)
        sys.exit()

    ################# colorPlateNumber = 2 ###################
    # canny edge image
    kernel = 'sobel'
    thresLow=20
    thresHigh=40

    cannyImage = cannyEdge(imRGB, thresHigh=thresHigh, thresLow=thresLow, sigma=1.0, version='edge', kernel=kernel)
    if colorPlateNum == 2:
        plt.imshow(cannyImage)
        plt.pause(3)
        colorImSave('edge_'+kernel+'.png', cannyImage)
        sys.exit()

    # random number seed
    np.random.seed(29645)

    ################ colorPlateNumber = 3 ####################
    # orientation of paint brush strokes
    if colorPlateNum == 3:
        # fixed theta=45
        theta = pi*3 / 4
    else:
        ################# colorPlateNumber = 4 #################
        # random theta, but still fixed
        theta = 2 * pi * np.random.rand(1,1)[0][0]

    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]

    # set radius of paint brush and half length of drawn lines
    rad = 2.0
    halfLen = 8

    # set up x,y coordinate images, and canvas
    # x and y is used for location indication, and canvas is used for color fillings
    [x, y] = np.meshgrid(np.array([i+1 for i in range(int(sizeIm[1]))]),
                         np.array([i+1 for i in range(int(sizeIm[0]))]))
    canvas = np.zeros((sizeIm[0], sizeIm[1], 3))
    canvas.fill(-1) # initially mark the canvas with a value out of range
    # it is used to indicate the pixels not painted

    # set vector from center to one end of the stroke
    delta = np.array([cos(theta), sin(theta)])

    time.time()
    time.clock()

    ###################### colorPlateNum = 5 ####################
    # thetas in [0, 45, 90, 135]
    if colorPlateNum == 5:
        thetas = cannyEdge(imRGB, sigma=1.0, version='theta1')
    ###################### colorPlateNum = 6 ####################
    elif colorPlateNum == 6:
        thetas = cannyEdge(imRGB, sigma=1.0, version='theta2')
    ###################### colorPlateNum = 7 ####################
    elif colorPlateNum == 7:
        imRGBRand = (2 * np.random.rand(sizeIm[0], sizeIm[1], 3)-1) * 15/255.0
        imRGB += imRGBRand
        thetas = cannyEdge(imRGB, sigma=1.0, version='theta2')
        thetaRand = (2 *np.random.rand(sizeIm[0], sizeIm[1])-1) * 15 / 180.0 * np.pi
        thetas += thetaRand


    k = 1 # k is used for recording the number of strokes being printed out
    while len(np.where(canvas==-1)[0]) != 0:
        # find a negative pixel
        # randomly select stroke center
        cntr = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1], sizeIm[0]])) + 1
        cntr = np.amin(np.vstack((cntr, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        # grab colour from image at center position of the stroke
        color = np.reshape(imRGB[cntr[1]-1, cntr[0]-1, :], (3,1))

        # add the stroke to the canvas
        nx, ny = (sizeIm[1], sizeIm[0])
        length1, length2 = (halfLen, halfLen)

        if colorPlateNum in [5, 6, 7]:
            # recompute theta and delta for endPoint
            theta = thetas[cntr[1]-1, cntr[0]-1]
            delta = np.array([cos(theta+np.pi/2), sin(theta++np.pi/2)])
            endPoint1, endPoint2 = findEndPoint(cannyImage, cntr, delta, halfLen, sizeIm)
        else:
            endPoint1 = cntr-delta*length2
            endPoint2 = cntr+delta*length1


        canvas = paintStroke(canvas, x, y, endPoint1, endPoint2, color, rad)
        print 'stroke', k
        k += 1


    print 'Done !!!'
    time.time()
    canvas[canvas < 0] = 0.0
    plt.clf()
    plt.axis('off')
    plt.imshow(canvas)
    plt.pause(3)
    if colorPlateNum == 3:
        colorImSave('output3.png', canvas)
    elif colorPlateNum == 4:
        colorImSave('output4.png', canvas)
    elif colorPlateNum == 5:
        colorImSave('output5.png', canvas)
    elif colorPlateNum == 6:
        colorImSave('output6.png', canvas)
    elif colorPlateNum == 7:
        colorImSave('output7.png', canvas)
