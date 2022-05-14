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

    testudo_img = cv2.imread('testudo.png')

    testudo_img_gr = cv2.cvtColor(testudo_img,cv2.COLOR_BGR2GRAY)


    tl,tb = np.shape(testudo_img_gr)
    print("Shape of Testudo:",tl,",",tb)

    if(tl>tb):

        testudo_img_gr = cv2.resize(testudo_img_gr,(tl,tl),interpolation=cv2.INTER_AREA)

        testudo_img = cv2.resize(testudo_img,(tl,tl),interpolation=cv2.INTER_AREA)

    if(tb>tl):

        testudo_img_gr = cv2.resize(testudo_img_gr,(tb,tb),interpolation=cv2.INTER_AREA)

        testudo_img = cv2.resize(testudo_img,(tb,tb),interpolation=cv2.INTER_AREA)

    
    tl,tb = np.shape(testudo_img_gr)
    print("Shape of image:",tl,",",tb)

    #out = cv2.VideoWriter('vr_testudo.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(w,h))

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

            # corner_counter = 0
            # for i in corners:

            #     x,y = i.ravel()
            #     if(corner_counter==0):
            #         col = (0,255,0)
            #     if(corner_counter==1):
            #         col = (255,0,0)
            #     if(corner_counter==2):
            #         col = (0,0,255)
            #     if(corner_counter==3):
            #         col = (0,127,127)
            #     corner_counter+=1
            #     cv2.circle(frame,(x,y),10,col,-1)

            H = Homography(corners,tl,tb)

            tag_img = id_extractor(H,tl,tb,g)
            id_val,rotations = id_detect(tag_img)

            print("Tag id:",id_val,"frame num:",counter)

            frame = image_imposer(frame,testudo_img,H,tl,tb,rotations)
            
            cv2.imshow("Frame_Displayer", frame)
            #out.write(frame)

            if (counter==10):
                cv2.imwrite("Isolated patch.png",cam_april)
                gray_frame = g
                col_frame = frame
            counter+=1

            if cv2.waitKey(30) & 0XFF == ord('q'):break

        else:break

    cap.release()
    #out.release()

    cv2.destroyAllWindows()

    return gray_frame,col_frame
##############################################################################################################################

"""
Perform FFT
"""

def fft_noise_removal(gray_img):

    ftimg = np.fft.fft2(gray_img)
    ftimg = np.fft.fftshift(ftimg)

    l,b = np.shape(ftimg)

    cx = int(l/2)
    cy = int(b/2)

    for i in range(250):

        for j in range(250):

            ftimg[cx-i][cy-j] = 0
            ftimg[cx-i][cy+j] = 0
            ftimg[cx+i][cy+j] = 0
            ftimg[cx+i][cy-j] = 0

    invftimg = np.fft.ifftn(ftimg)

    return ftimg,invftimg

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
ID Extraction
"""
def id_extractor(H,tl,tb,gray_img):

    blank_img = np.zeros((tl,tb))

    for i in range(tl):
        for j in range(tb):

            x = np.array([i,j,1]).T
            p = np.matmul(H,x)
            p = np.divide(p,p[-1])
            blank_img[i,j] = gray_img[int(p[1]),int(p[0])]
    
    _,blank_img = cv2.threshold(blank_img,200,255,cv2.THRESH_BINARY)

    return blank_img

##############################################################################################################################

"""
ID value and orientation
"""

def id_detect(id_img):

    l,b = np.shape(id_img)
    pl = int(l/8)
    pb = int(b/8)
    
    oc3 = id_img[int(6*pl),int(6*pb)]
    rotation_counter = 0

    while(oc3!=255 and rotation_counter<4):
        id_img = cv2.rotate(id_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotation_counter+=1
        oc3 = id_img[int(6*pl),int(6*pb)]

    print(oc3)
    b1 = (id_img[int(3.5*pl),int(3.5*pb)]==255)
    b2 = (id_img[int(3.5*pl),int(4.5*pb)]==255)
    b3 = (id_img[int(4.5*pl),int(4.5*pb)]==255)
    b4 = (id_img[int(4.5*pl),int(3.5*pb)]==255)

    id = b1*np.power(2,0) + b2*np.power(2,1) + b3*np.power(2,2) + b4*np.power(2,3)

    return id,rotation_counter

##############################################################################################################################
"""
Image imposition
"""
def image_imposer(color_img,imposer_img,H,tl,tb,rotations):

    for i in range(rotations):

        imposer_img = cv2.rotate(imposer_img,cv2.ROTATE_90_CLOCKWISE)

    for i in range(tl):
        for j in range(tb):

            x = np.array([i,j,1]).T
            p = np.matmul(H,x)
            p = np.divide(p,p[-1])
            color_img[int(p[1]),int(p[0])] = imposer_img[i,j]

    return color_img

##############################################################################################################################
"""
Main function
"""
def main():

    ##### Read videos #####
    vid_name = '1tagvideo.mp4'

    ##### Resize testudo image as a square #####

    testudo_img = cv2.imread('testudo.png')

    testudo_img_gr = cv2.cvtColor(testudo_img,cv2.COLOR_BGR2GRAY)


    tl,tb = np.shape(testudo_img_gr)
    print("Shape of Testudo:",tl,",",tb)

    if(tl>tb):

        testudo_img_gr = cv2.resize(testudo_img_gr,(tl,tl),interpolation=cv2.INTER_AREA)

        testudo_img = cv2.resize(testudo_img,(tl,tl),interpolation=cv2.INTER_AREA)

    if(tb>tl):

        testudo_img_gr = cv2.resize(testudo_img_gr,(tb,tb),interpolation=cv2.INTER_AREA)

        testudo_img = cv2.resize(testudo_img,(tb,tb),interpolation=cv2.INTER_AREA)

    testudo_img = cv2.cvtColor(testudo_img,cv2.COLOR_BGR2RGB)
    tl,tb = np.shape(testudo_img_gr)
    print("Shape of image:",tl,",",tb)

    ##### Extract 11th Frame,threshold it, perform fft operation, and isolate april code patch #####

    gr_img,col_img = vid_read(vid_name)
    col_img = cv2.cvtColor(col_img,cv2.COLOR_BGR2RGB)
    _,g_iso = cv2.threshold(gr_img,180,255,cv2.THRESH_BINARY)


    fft,invftimg = fft_noise_removal(g_iso)

    plt.figure("Color, isolated, fft, and patch")
    plt.subplot(2,2,1)
    plt.axis("off")
    plt.title("Grayscale Image")
    plt.imshow(gr_img,cmap="gray")

    plt.subplot(2,2,2)
    plt.axis("off")
    plt.title("Isolated page")
    plt.imshow(g_iso,cmap="gray")

    plt.subplot(2,2,3)
    plt.axis("off")
    plt.title("Fourier Transform(tapered)")
    plt.imshow(np.log(np.abs(fft)),cmap="gray")

    plt.subplot(2,2,4)
    plt.axis("off")
    plt.title("Fourier Transform of Image")
    plt.imshow(np.abs(invftimg),cmap="gray")
    
    plt.show()
    ##### Image morphology and corner detection #####
    cam_april = g_iso.copy()
    cv2.floodFill(cam_april,None,(0,0),255)
    dino = power_ranger(cam_april)
    cam_april = cv2.bitwise_and(cam_april,dino)

    corners = cv2.goodFeaturesToTrack(cam_april,4,0.01,80)
    corners = np.int0(corners)

    corners = corner_selector(corners)

    ##### Homography #####
    H = Homography(corners,tl,tb)

    print("Homography Matrix:",H)

    ##### Tag Id detection and image superimposition#####
    tag_img = id_extractor(H,tl,tb,gr_img)

    id_val,rotations = id_detect(tag_img)

    print("TAG id:",id_val)

    col_img = image_imposer(col_img,testudo_img,H,tl,tb,rotations)

    plt.figure("Tag")
    plt.axis("off")
    plt.imshow(tag_img,cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()