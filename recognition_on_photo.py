import mtcnn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from keras.models import load_model
import os
import cv2


class PhotoFaceRecogniser:
    model = load_model('CustomVggModel.h5')
    face_detector = mtcnn.MTCNN()
    BASE_DIR = "faces_base"

    def get_names(self):
        names_list = []
        for name in os.listdir(self.BASE_DIR):
            names_list.append(name)
        names_list.sort()
        return names_list

    def get_files_names(self, directory):
        files_names = []
        for filename in os.listdir(directory):
            files_names.append(str(filename))
        return files_names

    def detect_faces(self, directory):
        face_detector = mtcnn.MTCNN()
        faces_placement, faces_list, files_name = [], [], []
        for filename in os.listdir(directory):
            print(filename, ":")
            photo = plt.imread(f'{directory}/{filename}')
            face_locations = face_detector.detect_faces(photo)
            faces_on_photo, face_coordinates = [], []
            for face_location in face_locations:
                x1, y1, width, height = face_location['box']
                x2, y2 = x1 + width, y1 + height
                face = photo[y1:y2, x1:x2]
                face = Image.fromarray(face).resize((224, 224))
                face = np.array(face.getdata()).reshape((224, 224, 3))
                faces_on_photo.append(face)
                face_coordinates.append([(x1, y1), (x2, y2)])
            faces_list.append(faces_on_photo)
            faces_placement.append(face_coordinates)
        return faces_placement, faces_list

    def recognise(self, faces_list):
        predictions = []
        for faces_on_photo in faces_list:
            x = np.asarray(faces_on_photo)
            frame_predictions = np.ndarray.tolist(self.model.predict(x))
            predictions.append(frame_predictions)
        return predictions

    def show_results(self, test_directory, predictions, faces_placement):
        FRAME_THICKNESS = 2
        FONT_THICKNESS = 2
        i = 0
        files_names = self.get_files_names(test_directory)
        names_list = self.get_names()
        for image_predictions in predictions:
            j = 0
            filename = files_names[i]
            image = PIL.Image.open(f'{test_directory}/{filename}')
            image = image.convert("RGB")
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for prediction in image_predictions:
                person_name = names_list[prediction.index(max(prediction))]
                print(files_names[i], " - ", person_name)
                if person_name != "unknown":
                    print(f'Match found: {person_name}')
                    color = [0, 255, 0]
                    font_color = (0, 0, 0)
                else:
                    color = [0, 0, 255]
                    font_color = (255, 255, 255)
                face_cords = faces_placement[i][j]
                cv2.rectangle(image, face_cords[0], face_cords[1], color, FRAME_THICKNESS)
                top_left = (face_cords[0][0], face_cords[1][1])
                bottom_right = (face_cords[1][0], face_cords[1][1] + 20)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, person_name, (face_cords[0][0] + 10, face_cords[1][1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, FONT_THICKNESS)
                j += 1
            cv2.imshow(filename, image)
            i += 1

            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyWindow(filename)


directory = "unknown_faces"
face_recogniser = PhotoFaceRecogniser()
face_locations, detected_faces = face_recogniser.detect_faces(directory)
predictions = face_recogniser.recognise(detected_faces)
face_recogniser.show_results(directory, predictions, face_locations)

