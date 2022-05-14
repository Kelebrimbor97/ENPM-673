import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import cv2

from random import sample
###########################################################################################

def img_reader(path):

    img_set = sorted(glob.glob(path+'*.png'))
    img_set_col = [cv2.imread(i) for i in img_set]
    img_set_gray = [cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in img_set_col]

    return img_set_col,img_set_gray

###########################################################################################

def param_reader(path):

    if(path == 'Datasets/data-20220410T231452Z-001/data/curule/'):

        cam0=np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0, 0, 1]])
        cam1=np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15], [0, 0, 1]])
        doffs=0
        baseline=88.39
        width=1920
        height=1080
        ndisp=220
        vmin=55
        vmax=195

        return cam0,cam1,doffs,baseline,width,height,ndisp,vmin,vmax

    elif(path == 'Datasets/data-20220410T231452Z-001/data/pendulum/'):

        cam0=np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0, 0, 1]])
        cam1=np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0, 0, 1]])
        doffs=0
        baseline=537.75
        width=1920
        height=1080
        ndisp=180
        vmin=25
        vmax=150

        return cam0,cam1,doffs,baseline,width,height,ndisp,vmin,vmax

    elif(path == 'Datasets/data-20220410T231452Z-001/data/octagon/'):

        cam0=np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0, 0, 1]])
        cam1=np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0, 0, 1]])
        doffs=0
        baseline=221.76
        width=1920
        height=1080
        ndisp=100
        vmin=29
        vmax=61

        return cam0,cam1,doffs,baseline,width,height,ndisp,vmin,vmax

    else:
        return None

###########################################################################################

def feature_matching(img1,img2):

    orb = cv2.ORB_create(nfeatures=5000)

    k1,des1 = orb.detectAndCompute(img1,None)
    k2,des2 = orb.detectAndCompute(img2,None)

    index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
    search_params = {}

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.64*n.distance:
            good.append(m)

    # img3 = cv2.drawMatchesKnn(img1,k1,img2,k2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show()

    left_points = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 2)

    right_points = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    return left_points,right_points

###########################################################################################

def EstimateFundamentalMatrix(left_pts,right_pts):
    A = []

    for i in range(len(left_pts)):

        xi_left = np.array([[left_pts[i][0],left_pts[i][1],1]])
        xi_right = np.array([[right_pts[i][0],right_pts[i][1],1]])

        mult_vect = np.reshape(np.matmul(xi_left.T,xi_right),9)
        A.append(mult_vect)

    A = np.array(A)
    #print(np.shape(A))

    U,S,Vt = np.linalg.svd(A)
    F = np.reshape(Vt[-1],(3,3))
    #print("Fundamental Matrix pre rank correction:\n",F,"\nrank=",np.linalg.matrix_rank(F))
    U,S,Vt = np.linalg.svd(F)
    S[-1] = 0
    F = np.matmul(U*S,Vt)
    #F = np.divide(F,F[-1,-1])
    #print("\n\nFundamental Matrix after rank correction:\n",F,"\nrank=",np.linalg.matrix_rank(F))

    return F

###########################################################################################

def EstimateEssentialMatrix(F,K1,K2):

    E = np.matmul(np.matmul(K2.T,F),K1)

    U,S,Vt = np.linalg.svd(E)
    S[-1] = 0
    E = np.matmul(U*S,Vt)

    #print("\nEssential Matrix:\n",E)

    return E

###########################################################################################

def EstimateExtrinsicParams(E):

    U,S,Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    c1 = U[:,2]
    c2 = np.multiply(U[:,2],-1)
    c3 = c1
    c4 = c2
    c_set = [c1,c2,c3,c4]

    r1 = np.matmul(np.matmul(U,W),Vt)
    r2 = r1
    r3 = np.matmul(np.matmul(U,W.T),Vt)
    r4 = r3
    r_set = [r1,r2,r3,r4]

    for i in range(len(r_set)):

        c_set[i] = c_set[i]*np.sign(np.linalg.det(r_set[i]))
        r_set[i] = r_set[i]*np.sign(np.linalg.det(r_set[i]))
    
    return c_set,r_set

###########################################################################################

def RANSAC(left_pts,right_pts):

    iters = 10000
    minimum_inliers = 8
    thresh = 0.001

    left_inliers = []
    right_inliers = []

    indexes = np.arange(len(left_pts))

    for i in range(iters):

        rand_indexes = sample(np.ndarray.tolist(indexes),8)
        left_tmp = left_pts[rand_indexes]
        right_tmp = right_pts[rand_indexes]

        #print("\n\nLeft tmp shape:",np.shape(left_tmp))

        left_tmp_norm, T_left = Normalise(left_tmp)
        right_tmp_norm, T_right = Normalise(right_tmp)

        #print(np.shape(left_tmp_norm))
        #print(np.shape(right_tmp_norm))

        F_norm = EstimateFundamentalMatrix(left_tmp_norm,right_tmp_norm)
        F_est = np.matmul(np.matmul(T_left.T,F_norm),T_right)

        #print("F_est shape:", np.shape(F_est))

        left_inliers_tmp = []
        right_inliers_tmp = []

        for j in range(len(left_pts)):

            left_pt_tmp = np.array([left_pts[j][0],left_pts[j][1],1])
            right_pt_tmp = np.array([right_pts[j][0],right_pts[j][1],1])
            est = np.matmul(np.matmul(right_pt_tmp,F_est),left_pt_tmp.T)

            if np.abs(est)<thresh:

                left_inliers_tmp.append(left_pts[j])
                right_inliers_tmp.append(right_pts[j])
        
        if len(left_inliers_tmp) > minimum_inliers:

            F_final = F_norm

            left_inliers = left_inliers_tmp
            right_inliers = right_inliers_tmp

            minimum_inliers = len(left_inliers_tmp)
    
    #F_final = np.divide(F_final,F_final[-1,-1])
    print("\nF by RANSAC:\n",F_final)

    return F_final, left_inliers, right_inliers

###########################################################################################

def Normalise(pts):

    u = pts[:,0]
    v = pts[:,1]
    #print("u shape",np.shape(u))
    X = np.ones((np.shape(pts)[0],np.shape(pts)[-1]+1))
    #print("X shape:",np.shape(X))

    X[:,0] = u[:]
    X[:,1] = v[:]


    u_mean = np.mean(u)
    v_mean = np.mean(v)

    u_tilde = np.subtract(u,u_mean)
    v_tilde = np.subtract(v,v_mean)

    s =  np.sqrt(2/(np.divide(np.add(np.sum(np.square(u_tilde)),np.sum(np.square(v_tilde))),len(u))))
    #print("s=",s)

    m1 = np.array([[s,0,0],[0,s,0],[0,0,1]])
    m2 = np.array([[1,0,-u_mean],[0,1,-v_mean],[0,0,1]])

    T_mat = np.matmul(m1,m2)
    x_hat = np.matmul(T_mat,X.T)
    x_hat = x_hat.T
    x_hat = x_hat[:,:2]
    #print("x_hat shape:",np.shape(x_hat))

    return x_hat,T_mat

###########################################################################################

def drawLines(img, lines, left_pts, right_pts):

    w = img.shape[1]

    # img_l = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    for r, pt_l, pt_r in zip(lines, left_pts, right_pts):


        color = (255,200,0)
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0]*w )/r[1] ])
        img_l = cv2.line(img, (x0,y0), (x1,y1), color, 1)
        img_l = cv2.circle(img_l, (int(pt_l[0]),int(pt_l[1])), 5, (0,0,255),-1)
        img_l = cv2.circle(img_l, (int(pt_r[0]),int(pt_r[1])), 5, (0,0,255),-1)

    return img_l

###########################################################################################

def EpipolarPlotter(F_mat, left_pts, right_pts, img_left, img_right):

    left_lines = cv2.computeCorrespondEpilines(right_pts.reshape(-1,1,2),2,F_mat)
    right_lines = cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2),2,F_mat)

    left_lines = left_lines.reshape(-1,3)
    right_lines = right_lines.reshape(-1,3)

    left_img = img_left.copy()
    right_img = img_left.copy()

    left_img = drawLines(left_img, left_lines, left_pts, right_pts)
    right_img = drawLines(right_img, right_lines, left_pts, right_pts)

    return left_img, right_img

###########################################################################################

def warper(img_left,img_rght,F,left_pts,right_pts):

    left_new = np.reshape(left_pts,(-1,1,2))
    rght_new = np.reshape(right_pts,(-1,1,2))

    _, H_left, H_right = cv2.stereoRectifyUncalibrated(left_new,rght_new,F,(img_left.shape[1],img_left.shape[0]))

    img_warped_left = cv2.warpPerspective(img_left, H_left, (img_left.shape[1],img_left.shape[0]))
    img_warped_rght = cv2.warpPerspective(img_rght, H_right, (img_rght.shape[1],img_left.shape[0]))

    F_new = np.linalg.inv(H_right).T.dot(F).dot(np.linalg.inv(H_left))

    img_epi_left, img_epi_right = EpipolarPlotter(F_new, np.array(left_pts), np.array(right_pts), img_warped_left, img_warped_rght)

    plt.figure()

    plt.subplot(1,2,1)
    plt.axis("off")
    plt.imshow(img_epi_left)

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.imshow(img_epi_right)

    plt.show()

    return H_left, H_right, cv2.cvtColor(img_warped_left,cv2.COLOR_RGB2GRAY), cv2.cvtColor(img_warped_rght,cv2.COLOR_RGB2GRAY)

###########################################################################################

def GenerateDisparity(img_left, img_right, F_mat,baseline, ndisp):

    f = F_mat[0,0]

    blank_img = np.zeros_like(img_left,dtype=np.uint8)

    block_size = 3

    for i in range(block_size, img_left.shape[0], 2*block_size):

        x = np.repeat(np.arange(start=i - block_size, stop = i + block_size), 2*block_size)

        for j in range(block_size, int(img_left.shape[1]), 2*block_size):

            y = np.tile(np.arange(start = j -block_size, stop = j + block_size), 2*block_size)
            block_left = img_left[x,y]

            min_val = np.inf
            index = 1

            for k in range(max(block_size, j-90), j , block_size):

                z = np.tile(np.arange(start=k-block_size, stop=k + block_size), 2*block_size)

                block_right = img_right[x,z]

                ssd = np.sum(np.square(np.subtract(block_left,block_right)))

                if(ssd<min_val):

                    index = k
                    min_val = ssd

            blank_img[x,y] = np.uint8(np.abs(index-j))

    
    min_disp = np.min(blank_img)
    max_disp = np.max(blank_img)

    print("min_disp_val:", min_disp)
    print("max disp_val:", max_disp)

    depth_img = np.zeros((img_left.shape[0], img_left.shape[1]))

    scaler = ndisp/(max_disp-min_disp)

    disp_final = np.subtract(blank_img,min_disp)
    disp_final = np.uint8(np.multiply(disp_final, scaler))
    disp_final = cv2.cvtColor(disp_final,cv2.COLOR_GRAY2BGR)
    disp_final = cv2.fastNlMeansDenoisingColored(disp_final,None,55,55,7,21)
    disp_final = cv2.cvtColor(disp_final,cv2.COLOR_BGR2GRAY)
    disp_final = cv2.medianBlur(disp_final,9)
    disp_final = np.uint8(disp_final)
    depth_img = np.multiply(1/blank_img, baseline*f)

    plt.figure("Disparity map")
    plt.axis("off")
    plt.imshow(disp_final, cmap="plasma")

    plt.figure("Depth Map")
    plt.axis("off")
    plt.imshow(depth_img, cmap="plasma")

    plt.show()
    
###########################################################################################

def main():

    path = 'Datasets/data-20220410T231452Z-001/data/pendulum/'
    imgs_color, imgs_gray = img_reader(path)
    
    cam0,cam1,doffs,baseline,width,height,ndisp,vmin,vmax = param_reader(path)
    left_feats,right_feats = feature_matching(imgs_gray[0],imgs_gray[1])

    print("left_feats max=",np.max(left_feats[:,0]),np.max(left_feats[:,1]),"\nright_feats shape=",np.shape(right_feats),"\nimg shape",np.shape(imgs_gray[0]))

    left_norm, T_left = Normalise(left_feats)
    right_norm, T_right = Normalise(right_feats)
    F_ls = EstimateFundamentalMatrix(left_norm,right_norm)
    F_ls = np.matmul(np.matmul(T_left.T,F_ls),T_right)

    #F_ls = np.divide(F_ls,F_ls[-1,-1])
    print("\nF by Least squares:\n",F_ls,"\nshape:",np.shape(F_ls))

    F_RANSAC,left_inliers,right_inliers = RANSAC(left_feats, right_feats)
    #F_RANSAC = np.divide(F_RANSAC,F_RANSAC[-1,-1])
    print("\nshape:",np.shape(F_RANSAC),"rank:",np.linalg.matrix_rank(F_RANSAC))

    E = EstimateEssentialMatrix(F_RANSAC,cam0,cam1)
    print("\nEssential Matrix:\n",E)

    c_set, r_set = EstimateExtrinsicParams(E)

    print("\nExtrinsic Params:\n",c_set,"\n",r_set)


    H_left,H_right, img_left_w, img_right_w = warper(cv2.cvtColor(imgs_color[0],cv2.COLOR_BGR2RGB),cv2.cvtColor(imgs_color[1],cv2.COLOR_BGR2RGB),F_RANSAC,left_inliers,right_inliers)
    print("\nLeft Homography:\n",H_left)
    print("\nRight Homography:\n",H_right)

    GenerateDisparity(img_left_w, img_right_w, F_RANSAC, baseline, ndisp)

if __name__ == '__main__':
    main()