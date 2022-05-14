import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import cv2

########################################################################################################
"""
Function that reads images from the given path and returns sets of color and corresponding gray images
"""

def image_reader(path):

	img_set_gray = []
	img_set_color = []

	img_set_color = [cv2.imread(file) for file in sorted(glob.glob(path+'*.png'))]
	img_set_gray = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in img_set_color]

	return img_set_color,img_set_gray

########################################################################################################
"""
Vanilla histogram equalization
"""
def hist_eq_norm(img_gray):

    b,a = np.shape(img_gray)
    hist_img = img_gray.copy()
    bright_vect = np.zeros(256)

    for i in range(b):

        for j in range(a):

            bright_vect[img_gray[i,j]] += 1

    cfd = np.divide(np.cumsum(bright_vect),(a*b))
    for i in range(b):

        for j in range(a):

            hist_img[i,j] = cfd[hist_img[i,j]] *255

    return bright_vect,cfd,hist_img

########################################################################################################
"""
Basic adaptive histogram equalizqtion
"""
def adaptive_hist_eqaualizatoin(img_gray):

    b,a = np.shape(img_gray)
    b1 = int(b/8)
    a1 = int(a/8)
    blank_img = np.zeros((b,a))

    for i in range(8):

        for j in range(8):

            patch = img_gray[i*b1:i*b1+b1,j*a1:j*a1+a1].copy()
            _,__,hist_patch = hist_eq_norm(patch)
            blank_img[i*b1:i*b1+b1,j*a1:j*a1+a1] = hist_patch

    return blank_img

########################################################################################################

"""
Contrast Limited Adaptive Histogram equalization
"""

def clahe_nk(img_gray):

    b,a = np.shape(img_gray)
    b1 = int(b/8)
    a1 = int(a/8)
    blank_img = np.zeros((b,a))

    for i in range(8):

        for j in range(8):

            patch = img_gray[i*b1:i*b1+b1,j*a1:j*a1+a1].copy()
            hist,__,___ = hist_eq_norm(patch)

            t = int(np.max(hist)*0.64)

            gather = 0

            for k in range(len(hist)):

                if(hist[k]>t):

                    vals = hist[k] - t
                    hist[k] = t
                    gather += vals
            
            gather = int(gather/256)
            hist = np.add(hist,gather)

            cfd = np.divide(np.cumsum(hist),(b1*a1))

            for x in range(b1):

                for y in range(a1):

                    patch[x,y] = cfd[patch[x,y]]*255

            blank_img[i*b1:i*b1+b1,j*a1:j*a1+a1] = patch

    return blank_img

########################################################################################################

def main():

    Parser = argparse.ArgumentParser(description='Histogram Equalization')
    Parser.add_argument('-p', default='adaptive_hist_data/', help='Path to the images to be equalized')

    Args = Parser.parse_args()
    path = Args.p

    img_set_color, img_set_gray = image_reader(path)

    l = len(img_set_color)

    ind_vect = np.arange(256)
    a,b = np.shape(img_set_gray[0])

    for i in range(l):

        blank_img = np.zeros((int(a*2),int(b*2)))
        save_str = "img_combined_"+str(i+1)
        bright_vect,cfd,hist_img = hist_eq_norm(img_set_gray[i])
        ada_hist_img = adaptive_hist_eqaualizatoin(img_set_gray[i])
        clahe_img = clahe_nk(img_set_gray[i])
        blank_img[:a,:b] = img_set_gray[i]
        blank_img[:a,b:] = hist_img
        blank_img[a:,:b] = ada_hist_img
        blank_img[a:,b:] = clahe_img
        if(i>=9):
            cv2.imwrite('Output/'+str(i+1)+'.png',blank_img)
        else:
            cv2.imwrite('Output/'+str(0)+str(i+1)+'.png',blank_img)

        plt.figure(i+1)

        plt.subplot(2,2,1)
        plt.axis("off")
        plt.title("Original Frame "+str(i+1))
        plt.imshow(img_set_gray[i],cmap="gray")

        plt.subplot(2,2,2)
        plt.axis("off")
        plt.title("Histogram equalized Frame "+str(i+1))
        plt.imshow(hist_img,cmap="gray")

        plt.subplot(2,2,3)
        plt.axis("off")
        plt.title("Adaptive Histogram equalized Frame "+str(i+1))
        plt.imshow(ada_hist_img,cmap="gray")

        plt.subplot(2,2,4)
        plt.axis("off")
        plt.title("Contrast Limited Adaptive Histogram equalized Frame "+str(i+1))
        plt.imshow(clahe_img,cmap="gray")

        plt.figure("Histogram and CFD for Frame "+str(i+1))
        
        plt.subplot(1,2,1)
        plt.title("Histogram of Frame "+str(i+1))
        plt.plot(ind_vect,bright_vect)
        
        plt.subplot(1,2,2)
        plt.title("CFD of Frame"+str(i+1))
        plt.plot(ind_vect,cfd)

        plt.show()

if __name__ == '__main__':
    main()