from face_roc import draw_on_frame , add_img2db , cheak_1face
import cv2

add_img2db('PATH_TO_IMG.png/jpg')


cap = cv2.VideoCapture(1)
opend = cap.isOpened()
while opend :
    _ , frame = cap.read()
    frame3 = cv2.convertScaleAbs(frame, alpha=1, beta=10)
    frame2 = draw_on_frame(frame.copy())
    cv2.imshow('Frame', frame2)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
