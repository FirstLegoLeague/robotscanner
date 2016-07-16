import cv2
import numpy as np

thresh = 230
blur = 7
kernel = np.ones((5,5),np.uint8)

def run():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('./samples/test3.mp4')
    lastFrame = None
    cv2.namedWindow("frame", 1);
    cv2.namedWindow("mask", 1);
    cv2.namedWindow("result", 1);
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame = imutils.resize(img, width=500)

        lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel,chan_a,chan_b = cv2.split(lab_image)

        b_channel = cv2.bitwise_not(chan_b)
        a_channel = cv2.add(chan_a,b_channel)
        a_channel = cv2.bitwise_not(a_channel)
        a_channel_inv = cv2.bitwise_not(a_channel)

        im_bw = cv2.threshold(a_channel_inv, thresh, 255, cv2.THRESH_BINARY)[1]
        im_bw = cv2.GaussianBlur(im_bw,(blur,blur),0)


        ret ,im_bw = cv2.threshold(im_bw,thresh,255,cv2.THRESH_BINARY)
        im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)

        im = cv2.bitwise_and(frame,frame,mask = im_bw)
        # dst = cv2.add(roi,im)

        cv2.imshow('frame',frame)
        cv2.imshow('mask',im_bw)
        cv2.imshow('result',im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()

run()
# static()
cv2.destroyAllWindows()