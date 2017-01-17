import sys
# from PIL import Image
import cv2
from math import pi, sin, cos, tan
import numpy as np


def cot(angle):
    return 1/tan(angle)


def projection(theta,phi):
        if theta<0.615:
            return projectTop(theta,phi)
        elif theta>2.527:
            return projectBottom(theta,phi)
        elif phi <= pi/4 or phi > 7*pi/4:
            return projectLeft(theta,phi)
        elif phi > pi/4 and phi <= 3*pi/4:
            return projectFront(theta,phi)
        elif phi > 3*pi/4 and phi <= 5*pi/4:
            return projectRight(theta,phi)
        elif phi > 5*pi/4 and phi <= 7*pi/4:
            return projectBack(theta,phi)


def projectLeft(theta,phi):
        x = 1
        y = tan(phi)
        z = cot(theta) / cos(phi)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Left",x,y,z)


def projectFront(theta,phi):
        x = tan(phi-pi/2)
        y = 1
        z = cot(theta) / cos(phi-pi/2)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Front",x,y,z)


def projectRight(theta,phi):
        x = -1
        y = tan(phi)
        z = -cot(theta) / cos(phi)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Right",x,-y,z)


def projectBack(theta,phi):
        x = tan(phi-3*pi/2)
        y = -1
        z = cot(theta) / cos(phi-3*pi/2)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Back",-x,y,z)


def projectTop(theta,phi):
        a = 1 / cos(theta)
        x = tan(theta) * cos(phi)
        y = tan(theta) * sin(phi)
        z = 1
        return ("Top",x,y,z)


def projectBottom(theta,phi):
        a = -1 / cos(theta)
        x = -tan(theta) * cos(phi)
        y = -tan(theta) * sin(phi)
        z = -1
        return ("Bottom",x,y,z)


# Convert coords in cube to image coords
# coords is a tuple with the side and x,y,z coords
# edge is the length of an edge of the cube in pixels
def cubeToImg(coords,edge):
    if coords[0]=="Left":
        (x,y) = (int(edge*(coords[2]+1)/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Front":
        (x,y) = (int(edge*(coords[1]+3)/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Right":
        (x,y) = (int(edge*(5-coords[2])/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Back":
        (x,y) = (int(edge*(7-coords[1])/2), int(edge*(3-coords[3])/2) )
    elif coords[0]=="Top":
        (x,y) = (int(edge*(3-coords[1])/2), int(edge*(1+coords[2])/2) )
    elif coords[0]=="Bottom":
        (x,y) = (int(edge*(3-coords[1])/2), int(edge*(5-coords[2])/2) )
    return (x,y)


# convert the in image to out image
def convert(imgIn,imgOut):
    inSize = imgIn.shape[0:2]
    outSize = imgOut.shape[0:2]
    edge = inSize[1]/4   # the length of each edge in pixels
    for i in xrange(inSize[1]):     # i - 1, j - 0
        for j in xrange(inSize[0]):
            pixel = imgIn[j,i,:]
            phi = i * 2 * pi / inSize[1]
            theta = j * pi / inSize[0]
            res = projection(theta,phi)
            (x,y) = cubeToImg(res,edge)
            #if i % 100 == 0 and j % 100 == 0:
            #   print i,j,phi,theta,res,x,y
            if x >= outSize[1]:
                #print "x out of range ",x,res
                x=outSize[1]-1
            if y >= outSize[0]:
                #print "y out of range ",y,res
                y=outSize[0]-1
            imgOut[y,x,:] = pixel


if __name__ == "__main__":
    imgIn = cv2.imread('Equi_Images/livingroom_1024x512.jpg', cv2.IMREAD_COLOR)
    img_height = 256
    imgIn = cv2.resize(imgIn, (img_height*2, img_height), interpolation=cv2.INTER_AREA)
    inSize = imgIn.shape[0:2]
    imgOut = np.zeros((inSize[1]/4*3, inSize[1], 3), dtype='uint8')
    convert(imgIn,imgOut)
    cv2.startWindowThread()
    # cv2.namedWindow("Face Feature Extraction", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Face Feature Extraction", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    # while True:
    #     cv2.imshow('Face Feature Extraction', np.vstack((imgIn, imgOut)))
    #     k = cv2.waitKey(1000) & 0xff
    #     if k == ord('q'):
    #         break
    cv2.imwrite('Output_Images/source_to_des_convert.jpg', np.vstack((imgIn, imgOut)))
    cv2.destroyAllWindows()