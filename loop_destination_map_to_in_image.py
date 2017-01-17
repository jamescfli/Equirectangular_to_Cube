import cv2
from PIL import Image
from math import pi,sin,cos,tan,atan2,hypot,floor
import numpy as np


# get x,y,z coords from out image pixels coords
# i,j are pixel coords
# face is face number
# edge is edge length
def out_img_to_xyz(i, j, face, edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face==0: # back
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face==1: # left
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face==2: # front
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face==3: # right
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face==4: # top
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face==5: # bottom
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)


# convert using an inverse transformation
def convert_back(imgIn, imgOut):
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]/4   # the length of each edge in pixels
    for i in xrange(outSize[0]):
        face = int(i/edge) # 0 - back, 1 - left 2 - front, 3 - right
        if face==2:
            rng = xrange(0,edge*3)
        else:
            rng = xrange(edge,edge*2)

        for j in rng:
            if j<edge:
                face2 = 4 # top
            elif j>=2*edge:
                face2 = 5 # bottom
            else:
                face2 = face

            (x,y,z) = out_img_to_xyz(i, j, face2, edge)
            theta = atan2(y,x) # range -pi to pi
            r = hypot(x,y)
            phi = atan2(z,r) # range -pi/2 to pi/2
            # source img coords
            uf = ( 2.0*edge*(theta + pi)/pi )
            vf = ( 2.0*edge * (pi/2 - phi)/pi)
            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi
            # Pixel values of four corners
            A = inPix[ui % inSize[0],np.clip(vi,0,inSize[1]-1)]
            B = inPix[u2 % inSize[0],np.clip(vi,0,inSize[1]-1)]
            C = inPix[ui % inSize[0],np.clip(v2,0,inSize[1]-1)]
            D = inPix[u2 % inSize[0],np.clip(v2,0,inSize[1]-1)]
            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[i,j] = (int(round(r)),int(round(g)),int(round(b)))


def convert_back_cv2_wrapper(in_img_cv2):    # both input and output in cv2 format
    cv2_im = cv2.cvtColor(in_img_cv2,cv2.COLOR_BGR2RGB)
    imgIn = Image.fromarray(cv2_im)
    inSize = imgIn.size
    imgOut = Image.new("RGB",(inSize[0],inSize[0]*3/4),"black")
    convert_back(imgIn, imgOut)
    out_img_cv2 = np.array(imgOut)      # auto convert to 512x1024
    # out_img_cv2 = out_img_cv2[:, :, ::-1].copy()    # RGB to BGR
    out_img_cv2 = cv2.cvtColor(out_img_cv2, cv2.COLOR_RGB2BGR)
    return out_img_cv2


if __name__ == "__main__":
    # PIL Image
    imgIn = Image.open('Equi_Images/livingroom_1024x512.jpg')       # 1024x512, width goes first
    inSize = imgIn.size
    imgOut = Image.new("RGB",(inSize[0],inSize[0]*3/4),"black")
    convert_back(imgIn, imgOut)
    imgOut.save('Output_Images/des_to_source_back_convert.jpg')
    imgOut.show()

    # OpenCV
    imgIn = cv2.imread('Equi_Images/livingroom_1024x512.jpg', cv2.IMREAD_COLOR)
    src_img_height = 256    # 256 ~ display, 512 ~ save
    imgIn = cv2.resize(imgIn, (src_img_height * 2, src_img_height), interpolation=cv2.INTER_AREA)
    imgOut = convert_back_cv2_wrapper(imgIn)
    cv2.startWindowThread()
    cv2.namedWindow("Face Feature Extraction", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face Feature Extraction", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow('Face Feature Extraction', np.vstack((imgIn, imgOut)))
        k = cv2.waitKey(1000) & 0xff
        if k == ord('q'):
            break
    cv2.imwrite('Output_Images/des_to_source_back_convert.jpg', np.vstack((imgIn, imgOut)))
    cv2.destroyAllWindows()

    # OpenCV get the top
    imgIn = cv2.imread('Equi_Images/livingroom_1024x512.jpg', cv2.IMREAD_COLOR)
    # src_img_height = 512    # 256 ~ display, 512 ~ save
    # imgIn = cv2.resize(imgIn, (src_img_height * 2, src_img_height), interpolation=cv2.INTER_AREA)
    imgOut = convert_back_cv2_wrapper(imgIn)
    img_top = imgOut[0:src_img_height / 2, (src_img_height / 2 * 2):(src_img_height / 2 * 3), :]
    cv2.imwrite('Output_Images/cube_top.jpg', img_top)