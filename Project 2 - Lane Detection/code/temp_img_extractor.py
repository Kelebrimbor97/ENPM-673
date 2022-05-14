import cv2
 
path = 'Output/problem2_op.mp4'

cap = cv2.VideoCapture(path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if(not cap.isOpened()):print("Error in opening video")

frame_counter =0

while(cap.isOpened()):

    ret, frame = cap.read()

    if ret:

        cv2.imshow("He bagh",frame)
        frame_counter+=1

        if(frame_counter==9):
            cv2.imwrite("Output/line_detection.png",frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):break

    else:break

cap.release()