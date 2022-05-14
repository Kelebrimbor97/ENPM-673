from re import L
import numpy as np
import matplotlib.pyplot as plt
import cv2

##############################################################################################################################
"""
Read video and if needed, store required output
"""
def vid_read(vid_name):
    cap = cv2.VideoCapture(vid_name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =cap.get(cv2.CAP_PROP_FPS)
    K = np.array([[1346.10059534175,0,932.163397529403],[0,1346.10059534175,654.898679624155],[0,0,1]])

    out = cv2.VideoWriter('ar_cube.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(w,h))

    if(not cap.isOpened()):print("Error in openeing video")

    counter = 0
    gray_frame = []
    col_frame = []

    
    while(cap.isOpened()):

        ret,frame = cap.read()
        if ret:

            g = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            _,iso_g = cv2.threshold(g,185,255,cv2.THRESH_BINARY)

            cam_april = iso_g.copy()
            cv2.floodFill(cam_april,None,(0,0),255)
            cam_april = power_ranger(cam_april)            

            corners = cv2.goodFeaturesToTrack(cam_april,4,0.01,125)
            corners = np.int0(corners)
            corners = corner_selector(corners)
    
            cam_april = cv2.cvtColor(cam_april,cv2.COLOR_GRAY2BGR)


            tl,tb = lb_calc(corners)
            H = Homography(corners,tl,tb)

            xw = world_coords(tl,tb)

            lamb = find_lambda(H,K)

            B = B_calc(lamb,H,K)

            P = P_calc(B,K)

            xc = []

            for i in xw:

                xc_temp = np.matmul(P,i)
                xc_temp = np.divide(xc_temp,xc_temp[-1])
                xc_temp = xc_temp[0:2]
                print("Temp xc:",xc_temp)
                xc.append(xc_temp)
            
            cube_maker(frame,corners,xc)
            #cv2.imshow("Frame_Displayer", frame)
            out.write(frame)

            if (counter==10):
                gray_frame = g
                col_frame = frame
            counter+=1

            if cv2.waitKey(30) & 0XFF == ord('q'):break

        else:break

    cap.release()
    out.release()

    cv2.destroyAllWindows()

    return gray_frame,col_frame

##############################################################################################################################

"""
Morphs image... MIGHTY MORPHIN' TIME!!!!! Dino Thunder was the best btw
"""
def power_ranger(cam_img):

    k1 = np.ones((25,25),np.uint8)
    k2 = np.ones((85,85),np.uint8)
    
    for i in range(4):
        dino = cv2.morphologyEx(cam_img,cv2.MORPH_CLOSE,k1)
        dino = cv2.morphologyEx(cam_img,cv2.MORPH_OPEN,k2)

    for i in range(10):
        cam_img1 = cv2.bitwise_and(cam_img,dino)
        cam_img1 = cv2.GaussianBlur(cam_img1,(21,21),cv2.BORDER_DEFAULT)

    return cam_img1

##############################################################################################################################
"""
Function to put corners in a specific order
"""
def corner_selector(corners):

    dist = [np.sum(corners[0]),np.sum(corners[1]),np.sum(corners[2]),np.sum(corners[3])]

    c1i = np.argmin(dist)

    c1 = corners[c1i]

    sqd = []

    for i in corners:

        ed = np.sqrt(np.sum(np.square(np.subtract(c1,i))))
        sqd.append(ed)

    c3i = np.argmax(sqd)
    c3 = corners[c3i]

    for i in corners:

        val = straight_line(c1,c3,i)
        if(val<0):
            c2 = i
        elif(val>0):
            c4 = i

    final_corners = np.array([c4,c3,c2,c1])

    return final_corners
##############################################################################################################################    
"""
straight line helper function for corner selection
"""
def straight_line(c1,c3,ci):

    x1,y1 = c1.ravel()
    x3,y3 = c3.ravel()

    a = x3-x1
    b = y3-y1

    xi,yi = ci.ravel()
    val = a*yi + x3*b - xi*b - a*y3

    return val
##############################################################################################################################

"""
world co-ordinate maker
"""

def world_coords(tl,tb):

    x1 = 0
    x2 = tl
    x3 = tl
    x4 = 0

    y1 = 0
    y2 = 0
    y3 = tb
    y4 = tb

    h = -1 * int((tl+tb)/2)

    cw1 = [x1,y1,h,1]
    cw2 = [x2,y2,h,1]
    cw3 = [x3,y3,h,1]
    cw4 = [x4,y4,h,1]

    cw_final = np.array([cw1,cw2,cw3,cw4])

    return cw_final

##############################################################################################################################

"""
Homography calculator.
"""
def Homography(ordered_corners,iml,imb):

    c1 = ordered_corners[0]
    c2 = ordered_corners[1]
    c3 = ordered_corners[2]
    c4 = ordered_corners[3]

    x1 = 0
    x2 = iml
    x3 = iml
    x4 = 0

    y1 = 0
    y2 = 0
    y3 = imb
    y4 = imb

    xp1,yp1 = c1.ravel()
    xp2,yp2 = c2.ravel()
    xp3,yp3 = c3.ravel()
    xp4,yp4 = c4.ravel()

    A =[[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],[0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],[-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],[0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],[-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],[0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],[-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],[0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]]
    U,sig,V = np.linalg.svd(A)

    H = np.reshape(V[8],(3,3))

    return H

##############################################################################################################################

"""
Lambda calculator
"""

def find_lambda(H,K):

    H_temp = H
    h1 = H_temp[:,0]
    h2 = H_temp[:,1]

    K_inv = np.linalg.inv(K)
    a = np.linalg.norm(np.dot(K_inv,h1))
    b = np.linalg.norm(np.dot(K_inv,h2))

    l = (2/(a+b))

    return l
##############################################################################################################################

"""
Length and breadth calculator
"""

def lb_calc(corners):

    c4 = corners[0]
    c3 = corners[1]
    c2 = corners[2]

    tl = np.sqrt(np.sum(np.square(np.subtract(c4,c3))))
    tb = np.sqrt(np.sum(np.square(np.subtract(c3,c2))))

    return int(tb),int(tl)

##############################################################################################################################

"""
Find ~B and B
"""
def B_calc(lamb,H,K):

    Bt = np.multiply(np.matmul(np.linalg.inv(K),H),lamb)
    
    Bt_det_s = np.sign(np.linalg.det(Bt))

    B = np.multiply(Bt,Bt_det_s)

    return B

##############################################################################################################################

"""
Find P, by first finding[R|t]
"""

def P_calc(B,K):

    r1 = B[:,0]
    r2 = B[:,1]
    r3 = np.cross(r1,r2)
    t = B[:,2]
    
    Rt = np.array([r1,r2,r3,t]).T

    print("dim Rt:", np.shape(Rt),"shape of K:",np.shape(K))
    
    P = np.matmul(K,Rt)

    return P

##############################################################################################################################

"""
Cube vsualizer
"""
def cube_maker(img,corners, xworld):

    for i in range(len(corners)):

        x,y = corners[i].ravel()
        if(i==0):
            col = (0,255,0)
        if(i==1):
            col = (255,0,0)
        if(i==2):
            col = (0,0,255)
        if(i==3):
            col = (0,127,127)
        cv2.circle(img,(x,y),10,col,-1)

    for i in range(len(xworld)):

        xw = xworld[i][0]
        yw = xworld[i][1]

        if(i==0):
            col = (0,255,0)
        if(i==1):
            col = (255,0,0)
        if(i==2):
            col = (0,0,255)
        if(i==3):
            col = (0,127,127)
        cv2.circle(img,(int(np.abs(xw)),int(np.abs(yw))),10,col,-1)

    for i in range(-1,len(corners)-1):

        xc1,yc1 = corners[i].ravel()

        xw1 = int(xworld[i][0])
        yw1 = int(xworld[i][1])

        
        xc2,yc2 = corners[i+1].ravel()

        xw2 = int(xworld[i+1][0])
        yw2 = int(xworld[i+1][1])

        if(i==(-1)):
            col = (51,153,255)
        if(i==0):
            col = (0,255,0)
        if(i==1):
            col = (255,0,0)
        if(i==2):
            col = (0,0,255)
        if(i==3):
            col = (0,127,127)
        cv2.line(img,(xc1,yc1),(xw1,yw1),col,4)
        cv2.line(img,(xc1,yc1),(xc2,yc2),col,4)
        cv2.line(img,(xw1,yw1),(xw2,yw2),col,4)

    

##############################################################################################################################

"""
Main function
"""
def main():

    ##### Set K Matrix ######
    K = np.array([[1346.10059534175,0,932.163397529403],[0,1346.10059534175,654.898679624155],[0,0,1]])

    ##### Read videos #####
    vid_name = '1tagvideo.mp4'

    ##### Extract 11th Frame and threshold it #####

    gr_img,col_img = vid_read(vid_name)
    #col_img = cv2.cvtColor(col_img,cv2.COLOR_BGR2RGB)
    _,g_iso = cv2.threshold(gr_img,180,255,cv2.THRESH_BINARY)



    ##### Image morphology and corner detection #####

    cam_april = g_iso.copy()
    cv2.floodFill(cam_april,None,(0,0),255)
    dino = power_ranger(cam_april)
    cam_april = cv2.bitwise_and(cam_april,dino)

    corners = cv2.goodFeaturesToTrack(cam_april,4,0.01,80)
    corners = np.int0(corners)

    corners = corner_selector(corners)

    tl,tb = lb_calc(corners)
    print("Tl:",tl,"TB:",tb)

    ##### Homography #####
    H = Homography(corners,tl,tb)
    
    xw = world_coords(tl,tb)

    lamb = find_lambda(H,K)
    print("Homography Matrix:",H)

    B = B_calc(lamb,H,K)

    P = P_calc(B,K)

    xc = []

    for i in xw:

        xc_temp = np.matmul(P,i)
        xc_temp = np.divide(xc_temp,xc_temp[-1])
        xc_temp = xc_temp[0:2]
        print("Temp xc:",xc_temp)
        xc.append(xc_temp)

    print("shape xc:",np.shape(xc))

    
    col_img = cv2.cvtColor(col_img,cv2.COLOR_BGR2RGB)
    #cube_maker(col_img,corners,xc)
    # cv2.imshow("Vertices",col_img)
    # cv2.imwrite("AR_cube.png",col_img)
    # cv2.waitKey(0) & 0XFF == ord('q')
    plt.figure()
    plt.axis("off")
    plt.imshow(col_img)
    plt.show()
    # plt.figure("Color, isolated, fft, and patch")
    # plt.subplot(2,2,1)
    # plt.axis("off")
    # plt.title("Grayscale Image")
    # plt.imshow(gr_img,cmap="gray")

    # plt.subplot(2,2,2)
    # plt.axis("off")
    # plt.title("Isolated page")
    # plt.imshow(g_iso,cmap="gray")
    
    # plt.show()

if __name__ == "__main__":
    main()