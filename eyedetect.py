from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import time
import imutils
import dlib
import cv2

#define a function to compute the EAR
def eye_aspect_ration(eye):
    #compute the euclidean distance between the verticak eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    #compute the euclidean distance between the horizontal eye landmarks
    c = dist.euclidean(eye[0], eye[3])

    #EAR
    ear = (A+B) / (2.0 * c)
    return ear

#construct the arguments to pass
import argparse
p=argparse.ArgumentParser()
#p.add_argument("-p", "--shape-predictor", required=True, help="path to the facial landmarks predictor")
p.add_argument("-v", "--video", type=str, default="", help="path to the video file")
args = vars(p.parse_args())

#define two important constants for the system
#1. ear thresh. threshold that ear must fall below and then rise above for blink to occur
#2. ear consecutive frame. number of consecutive times the ear must be below the threshold for a blink to be registered
EAR_Thresh = 0.32
EAR_Consec_frames = 5
counter = 0 
total = 0 

detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])
pred = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(pred)

#grab the facial landmarks for the right eye and the left eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#start video stream
print("[INFO] starting video streaming...")
vs = VideoStream(src=0).start()
fileStream=False
time.sleep(1.0)

#loop over the frames in the video stream
while True:
    #ensure that all frames in the buffer are processed
    if fileStream and not vs.more():
        break
    with open("name.txt") as f:
        for line in f:
            if line == "Unknown":
                print("[-*-] You Cannot Be Authenticated!!")
                cv2.putText(frame, "[-*-] AUTHENTICATION DINIED!: Unknown", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        f.close()    

    #grab the next frame and convert it to gray
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect the gray face
    rects = detector(gray, 0)

    #loop over each face and apply facial landmarks detection
    for rect in rects:
        #convert facial landmark(x,y) to a numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #extract the left and right eye coodinates and compuute the EEAR for both eyes
        lefteye = shape[lStart:lEnd]
        righteye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ration(lefteye)
        rightEAR = eye_aspect_ration(righteye)

        #Average the EAR together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
   
        #compute the convex hull for the left and right eye, then
	#visualize each of the eyes
        leftEyeHull = cv2.convexHull(lefteye)
        rightEyeHull = cv2.convexHull(righteye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #check to see if EAR is below the blink threshold and if so incriment the blink frame counter
        if ear < EAR_Thresh:
            counter += 1
        else:
            if counter >= EAR_Consec_frames:
                total += 1

            #reset the counter
            #counter = 0
            if total == 10:
                cv2.putText(frame, "ACCESS GRANTED!", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break
            counter = 0

        #Draw the total number of blinks on the frame along with the computed eye aspect ratio for the frame
        cv2.putText(frame, "AUTHENTICATING {}".format(line), (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, line, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
        #show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
 
#do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

