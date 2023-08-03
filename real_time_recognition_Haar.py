import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
from keras.models import load_model
import os
import cv2
#import mtcnn


class RealTimeFaceRecogniser:
    model = load_model('CustomVggModel.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # face_detector = mtcnn.MTCNN() #CNN detector
    BASE_DIR = "faces_base"
    FRAME_THICKNESS = 2
    FONT_THICKNESS = 2

    def get_names(self):
        names_list = []
        for name in os.listdir(self.BASE_DIR):
            names_list.append(name)
        names_list.sort()
        return names_list

    def recognise(self, faces_on_frame):
        x = np.asarray(faces_on_frame)
        if len(x) == 0:
            return []
        frame_predictions = np.ndarray.tolist(self.model.predict(x))
        return frame_predictions

    def start_recognition(self):
        print("Processing video...")
        self.video = cv2.VideoCapture(0)
        while True:
            ret, image = self.video.read()
            if image is not None:
                image = cv2.resize(image, (720, 480))
            else:
                print("Image is empty!")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_locations = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            # face_locations = self.face_detector.detect_faces(frame) #CNN detection
            faces_on_frame, face_coordinates = [], []
            for face_location in face_locations:
                x1, y1, width, height = face_location
                x2, y2 = x1 + width, y1 + height
                face = image[y1:y2, x1:x2]
                face = Image.fromarray(face).resize((224, 224))
                face = np.array(face.getdata()).reshape((224, 224, 3))
                faces_on_frame.append(face)
                face_coordinates.append([(x1, y1), (x2, y2)])
            predictions = self.recognise(faces_on_frame)
            names_list = self.get_names()
            i = 0
            if len(predictions) > 0:
                for prediction in predictions:
                    person_name = names_list[prediction.index(max(prediction))]
                    print(person_name, predictions)
                    if -2 < sorted(prediction)[-2]-max(prediction) < 2:
                        color = [255, 0, 0]
                        font_color = (255, 255, 255)
                        person_name = "Not shure"
                    elif person_name != "unknown":
                        print(f'Match found: {person_name}')
                        color = [0, 255, 0]
                        font_color = (0, 0, 0)
                    else:
                        color = [0, 0, 255]
                        font_color = (255, 255, 255)
                    face_cords = face_coordinates[i]
                    cv2.rectangle(image, face_cords[0], face_cords[1], color, self.FRAME_THICKNESS)
                    top_left = (face_cords[0][0], face_cords[1][1])
                    bottom_right = (face_cords[1][0], face_cords[1][1] + 20)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, person_name, (face_cords[0][0] + 10, face_cords[1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, self.FONT_THICKNESS)
                    i += 1
            cv2.imshow('Security camera', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        return 0


face_recogniser = RealTimeFaceRecogniser()
face_recogniser.start_recognition()




