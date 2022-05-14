import numpy as np
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

    while(cap.isOpened()):

        ret,frame = cap.read()

        if ret:
            frame_counter+=1

            out_frame = line_detection(frame)

            cv2.imshow("Feast your eyes",out_frame)
            out.write(out_frame)

            if cv2.waitKey(60) & 0XFF == ord('q'):break
            print("Processed Frame",frame_counter)

        else:break
    
    cap.release()
    out.release()

    cv2.destroyAllWindows()
########################################################################################################
"""
detect lines
"""

def line_detection(frame):

    frame_col = frame.copy()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    b,a = np.shape(frame_gray)
    cutter = int(b/2)+50
    frame_lane = frame_gray[cutter:,:]

    bx,ax = np.shape(frame_lane)
    blank_img = np.zeros_like(frame_lane)
    trapezium = np.array([[0,bx],[int(ax/2)-5,0],[int(ax/2)+5,0],[ax,bx]])
    cv2.fillConvexPoly(blank_img,trapezium,1)
    frame_lane = cv2.bitwise_and(frame_lane,frame_lane,mask=blank_img)
    _,frame_lane = cv2.threshold(frame_lane,125,255,cv2.THRESH_BINARY)

    edges = cv2.Canny(frame_lane,0,200)
    edges = cv2.GaussianBlur(edges,(3,3),cv2.BORDER_DEFAULT)
    lines1 = cv2.HoughLinesP(edges,1,np.pi/180,10,maxLineGap=0)
    lines2 = cv2.HoughLinesP(edges,1,np.pi/180,300,maxLineGap=200)

    for line in lines1:
        
        x1,y1,x2,y2 = line[0]
        d1 = np.linalg.norm(np.subtract(np.array([x1,y1]),np.array([x2,y2])))
        d2 = np.linalg.norm(np.array([8,8]))
        if((y2+cutter)>=0 and d1>d2): 
            cv2.line(frame_col,(x1,y1+cutter),(x2,y2+cutter),(0,0,255),2)

    for line in lines2:
        
        x1,y1,x2,y2 = line[0]
        if((y2+cutter)>=0): 
            cv2.line(frame_col,(x1,y1+cutter),(x2,y2+cutter),(0,255,0),8)

    return frame_col

########################################################################################################
def main():

    Parser = argparse.ArgumentParser(description='Histogram Equalization')
    Parser.add_argument('-p', default='whiteline.mp4', help='Path to the images to be equalized')
    Parser.add_argument('-o',default='Output/problem2_op.mp4', help='Path to the output folder where video will be written')

    Args = Parser.parse_args()
    path = Args.p
    out_path = Args.o

    vid_processor(path,out_path)


########################################################################################################

if __name__ == '__main__':
    main()