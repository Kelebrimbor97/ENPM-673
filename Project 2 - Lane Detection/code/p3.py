import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

########################################################################################################
"""
video reading function
"""
def vid_processor(path,out_path):

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =  cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))

    if(not cap.isOpened()):print("Error in opening video")

    frame_counter = 0
    old_lfit, old_rfit = None,None

    while(cap.isOpened()):

        ret,frame = cap.read()

        if ret:

            out_frame, left_fit_now, right_fit_now = lane_detection(frame,old_lfit,old_rfit)

            cv2.imshow("Feast your eyes",out_frame)
            out.write(out_frame)

            frame_counter+=1

            old_lfit = left_fit_now
            old_rfit = right_fit_now

            if cv2.waitKey(60) & 0XFF == ord('q'):break
            print("Processed Frame",frame_counter)

        else:break
    
    cap.release()
    out.release()

    cv2.destroyAllWindows()

########################################################################################################
def lane_detection(frame, old_left_fit, old_right_fit):

    frame_col = frame.copy()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cl1 = cv2.createCLAHE(clipLimit=5,tileGridSize=[16,16])
    frame_gray = cl1.apply(frame_gray)

    b,a = np.shape(frame_gray)

    blank_img = np.zeros_like(frame_gray)

    #trapezium = np.array([[0,b-64],[int(a/2)-175,int(2*b/3)],[int(a/2)+175,int(2*b/3)],[a,b-64]])
    trapezium = np.array([[-a,b+350],[int(a/2)-40,int(2*b/3)-40],[int(a/2)+125,int(2*b/3)-40],[a,b+350]]) #Order (0,0),(0,1),(1,1),(1,0)
    rect = np.array([[0,0],[0,b],[a,0],[a,b]])

    cv2.fillConvexPoly(blank_img,trapezium,1)
    frame_gray = cv2.bitwise_and(frame_gray,frame_gray,mask=blank_img)

    _,frame_thresh = cv2.threshold(frame_gray,200 ,255,cv2.THRESH_BINARY)
    frame_thresh = cv2.bilateralFilter(frame_thresh,11,75,75)
    H_mat,H_inv,top_view = top_down(frame_thresh)

    left_fit,right_fit,left_lane_locs,right_lane_locs,vis_data = fitter(top_view)
    out_top = cv2.cvtColor(top_view,cv2.COLOR_GRAY2BGR)

    nonzero = top_view.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_top[nonzeroy[left_lane_locs], nonzerox[left_lane_locs]] = [0,255, 0]
    out_top[nonzeroy[right_lane_locs], nonzerox[right_lane_locs]] = [100, 200, 255]     #Warped lane detection

    if(left_fit is None or len(nonzerox)>30000 ):
        left_fit = old_left_fit
    if(right_fit is None or len(nonzerox)>30000):
        right_fit = old_right_fit

    rects = vis_data[0]

    for i in rects:
        if(not left_fit is None):
            cv2.rectangle(out_top,(i[2],i[0]),(i[3],i[1]),(255,0,0),2)
        if(not right_fit is None):
            cv2.rectangle(out_top,(i[4],i[0]),(i[5],i[1]),(0,0,255),2)

    out_real = col_lane(frame,top_view,left_fit,right_fit,H_inv)

    left_radius, right_radius, d_center = rad_curve_lane_center(top_view,left_fit,right_fit,left_lane_locs,right_lane_locs)
    
    avg_turn_rad = np.mean([left_radius,right_radius])

    if(avg_turn_rad<80):
        str3 = 'Turn Right'
        col3 = (0,0,255)
    elif(avg_turn_rad<0):
        str3 = 'Turn Left'
        col3 = (0,0,255)
    else:
        str3 = 'Go Straight'
        col3 = (0,255,0)

    str1 ='Radii of lane curvature left:'+str(int(left_radius))+'m, right:'+str(int(right_radius))+'m'
    str2 ='Average turn: '+ str(avg_turn_rad)+'m'

    font = cv2.FONT_HERSHEY_DUPLEX
    org1 = (50,50)
    org2 = (50,100)
    org3 = (50,150)
    fontscale1 = 1

    color = (255,127,255)
    thickness = 2

    out_real = cv2.putText(out_real, str1, org1, font, fontscale1, color, thickness, cv2.LINE_AA)
    out_real = cv2.putText(out_real, str2, org2, font, fontscale1, color, thickness, cv2.LINE_AA)
    out_real = cv2.putText(out_real, str3, org3, font, fontscale1, col3, thickness, cv2.LINE_AA)
    
    return out_real, left_fit, right_fit

########################################################################################################
"""
Function to transform image to get a top-down view
"""
def top_down(img):

    bx,ax = np.shape(img)

    trapezium = np.array([[0,bx+350],[int(ax/2)-125,int(2*bx/3)-40],[int(ax/2)+125,int(2*bx/3)-40],[ax,bx+350]]) #Order (0,0),(0,1),(1,1),(1,0) 
    rect = np.array([[250,ax],[250,0],[250+bx,0],[250+bx,ax]]) #Order (0,0),(0,1),(1,1),(1,0)

    H_mat,_ = cv2.findHomography(trapezium,rect)
    H_inv,_ = cv2.findHomography(rect,trapezium)

    warped_img = cv2.warpPerspective(img,H_mat,(ax,bx),flags=cv2.INTER_LINEAR)

    return H_mat,H_inv,warped_img

########################################################################################################
"""
Fit a polynomial onto the lines using sliding window
"""
def fitter(img):
    
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    midpoint = int(histogram.shape[0]//2)
    quarter_point = int(midpoint//2)
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint+quarter_point-200:(midpoint+quarter_point+quarter_point-200)]) + midpoint + quarter_point -200

    nwindows = 10
    window_height = int(img.shape[0]/nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100
    
    minpix = 40
    left_lane_inds = []
    right_lane_inds = []
    
    rectangle_data = []

    for window in range(nwindows):
        
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit, right_fit = (None, None)
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histogram)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data

########################################################################################################
"""
Calculate Radius of Cuvature and Disctance from Lane Center 
"""

def rad_curve_lane_center(top_view, lfit, rfit, l_locs, r_locs):

    b,a = np.shape(top_view )

    ym_pix = 10.77/720
    xm_pix = 4.77/350

    left_lane_radius, right_lane_radius, center_dist = (0,0,0)

    y_eval = np.max(np.linspace(0,b-1,b))

    whitepixels = top_view.nonzero()
    white_y = np.array(whitepixels[0])
    white_x = np.array(whitepixels[1])

    left_x = white_x[l_locs]
    left_y = white_y[l_locs] 
    right_x = white_x[r_locs]
    right_y = white_y[r_locs]

    if not len(left_x)==0 and not len(right_x)==0:

        left_fit_cr = np.polyfit(left_y*ym_pix, left_x*xm_pix,2)
        right_fit_cr = np.polyfit(right_y*ym_pix, right_x*xm_pix,2)

        left_lane_radius = np.divide(np.power(np.sqrt(1 + np.square(2*left_fit_cr[0]*y_eval*ym_pix + left_fit_cr[1])),3),np.absolute(2*left_fit_cr[0]))
        right_lane_radius = np.divide(np.power(np.sqrt(1 + np.square(2*right_fit_cr[0]*y_eval*ym_pix + right_fit_cr[1])),3),np.absolute(2*right_fit_cr[0]))

    if not rfit is None and not lfit is None:
        car_center = top_view.shape[1]/2
        lfit_x = np.square(lfit[0]*b) + (lfit[1]*b) + lfit[2]
        rfit_x = np.square(rfit[0]*b) + (rfit[1]*b) + rfit[2]
        lane_center = (lfit_x + rfit_x)/2
        center_dist = (car_center - lane_center)*xm_pix
    
    return left_lane_radius,right_lane_radius,center_dist

########################################################################################################
"""
Detected lane put back onto original image
"""

def col_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
        
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    
    return result

########################################################################################################
def main():

    Parser = argparse.ArgumentParser(description='Histogram Equalization')
    Parser.add_argument('-p', default='challenge.mp4', help='Path to the images to be equalized')
    Parser.add_argument('-o',default='Output/problem3_op.mp4', help='Path to the output folder where video will be written')

    Args = Parser.parse_args()
    path = Args.p
    out_path = Args.o

    vid_processor(path,out_path)

########################################################################################################
if __name__ == '__main__':
    main()