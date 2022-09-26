
import numpy as np
import cv2
import dlib
import RPi.GPIO as GPIO
import argparse

from scipy.spatial import distance as dist

class BedControl:

    def __init__(self, video_channel=None, predict_path=None):
        self.video_channel = video_channel
        self.predict_path = predict_path
    
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))

        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 3

        self.COUNTER_LEFT = 0
        self.TOTAL_LEFT = 0

        self.COUNTER_RIGHT = 0
        self.TOTAL_RIGHT = 0

        self.COUNTER_BLINK = 0
        self.TOTAL_BLINK = 0

        self.PIN_MOTOR = 17
        self.PIN_MOTOR1 = 18
        self.PIN_EMERGENCY = 2
        self.PIN_TV = 3

        self.detector = dlib.get_frontal_face_detector()
        self.font = self.font

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.PIN_MOTOR1, GPIO.OUT)
        GPIO.setup(self.PIN_MOTOR, GPIO.OUT)
        GPIO.setup(self.PIN_EMERGENCY, GPIO.OUT)
        GPIO.setup(self.PIN_TV, GPIO.OUT)

    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear #Eye Aspect Ratio

    def reset_counter(self, reset):
        if reset == 'LEFT':
            self.TOTAL_RIGHT = 0
            self.COUNTER_RIGHT = 0 
            self.COUNTER_LEFT = 0 
            self.COUNTER_BLINK = 0
        elif reset == 'RIGHT':
            self.TOTAL_LEFT = 0 
            self.COUNTER_RIGHT = 0 
            self.COUNTER_LEFT = 0 
            self.COUNTER_BLINK = 0 
        elif reset == 'BLINK':
            self.TOTAL_LEFT = 0
            self.TOTAL_RIGHT = 0
            self.COUNTER_RIGHT = 0
            self.COUNTER_LEFT = 0 



    def main(self):
        cap = cv2.VideoCapture(self.video_channel)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            for rect in rects: 
                landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, rect).parts()])

                left_eye = landmarks[self.LEFT_EYE_POINTS]
                right_eye = landmarks[self.RIGHT_EYE_POINTS]

                ear_left = self.eye_aspect_ratio(left_eye)
                ear_right = self.eye_aspect_ratio(right_eye)
                ear_avg = (ear_left + ear_right) / 2

                leftEyeHull = cv2.convexHull(left_eye)
                rightEyeHull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

               
                if ear_left < self.EYE_AR_THRESH:
                    self.COUNTER_LEFT += 1
                else:
                    if self.COUNTER_LEFT >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL_LEFT += 1

                    self.reset_counter('LEFT')
                    GPIO.output(self.PIN_MOTOR, True) 

                if ear_right < self.EYE_AR_THRESH:
                    self.COUNTER_RIGHT += 1
                else:
                    if self.COUNTER_RIGHT >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL_RIGHT += 1
                    self.reset_counter('RIGHT')
                    GPIO.output(self.PIN_MOTOR, True) 

                if ear_avg < self.EYE_AR_THRESH:
                    self.COUNTER_BLINK += 1
                    if 20 < self.COUNTER_BLINK < 30:
                        GPIO.output(self.PIN_TV, False)
                        self.TOTAL_LEFT = 0
                        self.TOTAL_RIGHT = 0

                    if self.COUNTER_BLINK > 50:
                        GPIO.output(self.PIN_EMERGENCY, True)
                        self.reset_counter('BLINK')


                cv2.putText(frame, "EAR Left : {:.2f}".format(ear_left), (300, 30), self.font, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "EAR Right: {:.2f}".format(ear_right), (300, 60), self.font, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Wink Left : {} [{}]".format(self.TOTAL_LEFT, self.COUNTER_LEFT), (10, 30), self.font, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Wink Right: {} [{}]".format(self.TOTAL_RIGHT, self.COUNTER_RIGHT), (10, 60), self.font, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Blink: {} [{}]".format(self.TOTAL_BLINK, self.COUNTER_BLINK), (10, 90), self.font, 0.7, (0, 0, 255), 2)

            cv2.imshow("Blink Detection", frame)
            if 0xFF & cv2.waitKey(1) == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description="The study is based on the concepts and existing studies about image processing that covers facial recognition and object detection. The researchers use Raspberry Pi 4B for the programming work of the study and a good quality camera for facial recognition and object detection.")
parser.add_argument('-v','--video_channel', help='[0] for internal camera | [1] for external camera', default=0)
parser.add_argument('-p','--predict-path', help='Path of classes file', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    bc = BedControl(
        args['video_channel'],
        args['predict_path']
    )

    bc.main()