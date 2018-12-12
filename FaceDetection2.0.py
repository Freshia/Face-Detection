import cv2
import imutils
import numpy as np
import math
faceFinder=cv2.CascadeClassifier('F:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eyeFinder=cv2.CascadeClassifier('F:/Python/Lib/site-packages/cv2/data/haarcascade_eye.xml')
eyeglassesFinder=cv2.CascadeClassifier('F:/Python/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
cam=cv2.VideoCapture(0)

while(True):
    _,frame=cam.read()
    width= np.size(frame,1)
    height = np.size(frame,0)
    #resize the image to a width of 320
    if(width>320):
        scale = width/320
        scaledHeight = height / scale
        smallImg = imutils.resize(frame,320,scaledHeight)
        frame = smallImg

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #Equalize histogram to even out brightness and contrast

    equalized = cv2.equalizeHist(gray)
    faces = faceFinder.detectMultiScale(equalized,scaleFactor=1.1,minNeighbors=3)
    #draw rectangle over all faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        facewidth = (int)(w * scale)
        faceheight = (int)(h * scale)
        leftx = (int)(x*scale)
        topy = (int)(y*scale)

    #back to original size
    frame=imutils.resize(frame,width=width,height=height)
    #finding eye ROI
    for (x, y, w, h) in faces:

        faceROI = frame[topy:(topy+faceheight),leftx:leftx+facewidth]

        #eye search region for used cascades
        SX = 0.16
        SY = 0.26
        SW = 0.30
        SH = 0.28

        leftx = int(np.ceil(leftx+(facewidth*SX)))
        topy = int(np.ceil(topy+(faceheight * SY)))
        widthx = int(np.ceil(facewidth * SW))
        heighty = int(np.ceil(faceheight * SH))
        rightx = int(np.ceil((leftx+facewidth)-(facewidth*(1.0-SX - SW))))

        #defining right eye roi and left eye roi

        rightROI = frame[topy:topy+heighty,leftx:leftx+widthx]
        leftROI = frame[topy:topy + heighty, rightx:rightx + widthx]
        rightROIgray  = cv2.cvtColor(rightROI,cv2.COLOR_RGB2GRAY)
        leftROIgray = cv2.cvtColor(leftROI, cv2.COLOR_RGB2GRAY)

        #searching for eye in roi
        righteyes = eyeFinder.detectMultiScale(cv2.equalizeHist(rightROIgray))

        if not (np.asarray(righteyes).any()):
            righteyes = eyeglassesFinder.detectMultiScale(rightROIgray)

        for (ex, ey, ew, eh) in righteyes:
            cv2.rectangle(rightROI, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            rightCentreX = ex+(ew/2)
            rightCentreY = ey + (eh/2)

            lefteyes = eyeFinder.detectMultiScale(cv2.equalizeHist(leftROIgray))
            if not (np.asarray(lefteyes).any()):
                lefteyes = eyeglassesFinder.detectMultiScale(leftROIgray)

            for (ex, ey, ew, eh) in lefteyes:
                cv2.rectangle(leftROI, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                leftCentreX = ex + (ew/2)
                leftCentreY = ey + (eh/2)

                #if both face and eyes were detected

                print("both eyes are detected")
                #get center between the two eyes
                eyesCentreX = (rightCentreX+leftCentreX)/2
                eyesCentreY = (rightCentreY +leftCentreY)/2
                #finding angle between the two eyes
                dx = (rightCentreX - leftCentreX)
                dy = (rightCentreY-leftCentreY)
                len = math.sqrt(dx*dx+dy*dy)
                angle = math.atan2(dy,dx)*180/math.pi

                print(angle)

#left eye center should be at (0.16,0.14) of scaled face
                DESIRED_LEFT_EYE_X = 0.16
                DESIRED_RIGHT_EYE_X = (1.0 - 0.16)
                #get by how much we need to scale image
                DESIRED_FACE_WIDTH = 70
                DESIRED_FACE_HEIGHT = 70
                desiredLen = (DESIRED_RIGHT_EYE_X-0.16)
                scale = desiredLen * DESIRED_FACE_WIDTH / len
#now we're ready to transform some faces
                RotationMatrix = cv2.getRotationMatrix2D((eyesCentreX,eyesCentreY),angle=angle,scale=scale)
#shift eyes center to desired
                ex = DESIRED_FACE_WIDTH * 0.5 - eyesCentreX
                ey = DESIRED_FACE_HEIGHT *(0.14 - eyesCentreY)
                #rot_mat.at < double > (0, 2) += ex;
                #rot_mat.at < double > (1, 2) += ey;

#transform face to desired angle and size and background to default gray
                warped = np.array((DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH),dtype=np.float32)
                M = np.float32([[1, 0, 70], [0, 1, 70]])
                #print(warped.size)
                rows,cols = frame.shape[:2]


               # cv2.warpAffine(gray,warped, (rows,cols));




    cv2.imshow('frame', frame)
    cv2.imshow('gray', equalized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
