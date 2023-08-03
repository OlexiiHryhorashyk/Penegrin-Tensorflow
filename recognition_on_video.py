import mtcnn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
from keras.models import load_model
import os
import cv2


class VideoFaceRecogniser:
    model = load_model('CustomVggModel.h5')
    face_detector = mtcnn.MTCNN()
    BASE_DIR = "faces_base"

    def get_names(self):
        names_list = []
        for name in os.listdir(self.BASE_DIR):
            names_list.append(name)
        names_list.sort()
        return names_list

    def detect_faces(self, video_name):
        face_detector = mtcnn.MTCNN()
        faces_placement, faces_list = [], []
        print("Processing video...")
        self.video = cv2.VideoCapture(video_name)
        while True:
            ret, frame = self.video.read()
            if frame is not None:
                frame = cv2.resize(frame, (720, 480))
            else:
                print("Image is empty!")
                break
            face_locations = face_detector.detect_faces(frame)
            faces_on_frame, face_coordinates = [], []
            for face_location in face_locations:
                x1, y1, width, height = face_location['box']
                x2, y2 = x1 + width, y1 + height
                face = frame[y1:y2, x1:x2]
                face = Image.fromarray(face).resize((224, 224))
                face = np.array(face.getdata()).reshape((224, 224, 3))
                faces_on_frame.append(face)
                face_coordinates.append([(x1, y1), (x2, y2)])
            faces_list.append(faces_on_frame)
            faces_placement.append(face_coordinates)
        self.video.release()
        return faces_placement, faces_list

    def recognise(self, faces_list):
        predictions = []
        for faces_on_frame in faces_list:
            x = np.asarray(faces_on_frame)
            if len(x) == 0:
                predictions.append([])
                continue
            frame_predictions = np.ndarray.tolist(self.model.predict(x))
            predictions.append(frame_predictions)
        return predictions

    def write_video(self, file_path, frames, fps):
        w, h = frames[0].size
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        writer.release()

    def get_recognised_frames(self, video_name, predictions, faces_placement):
        FRAME_THICKNESS = 2
        FONT_THICKNESS = 2
        out_frames = []
        self.video = cv2.VideoCapture(video_name)
        i = 0
        names_list = self.get_names()
        for image_predictions in predictions:
            ret, image = self.video.read()
            if ret:
                image = cv2.resize(image, (720, 480))
            else:
                print("Image is empty!")
                continue
            image = Image.fromarray(image)
            image = image.convert("RGB")
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            j = 0
            if len(image_predictions) > 0:
                for prediction in image_predictions:
                    person_name = names_list[prediction.index(max(prediction))]
                    if person_name == "Trump":
                        print(f'Match found: {person_name}')
                        color = [0, 255, 0]
                        font_color = (0, 0, 0)
                    else:
                        color = [255, 0, 0]
                        font_color = (255, 255, 255)
                    try:
                        face_cords = faces_placement[i][j]
                    except IndexError:
                        print(faces_placement[i], " - ", prediction)
                        print("Index error!")
                    cv2.rectangle(image, face_cords[0], face_cords[1], color, FRAME_THICKNESS)
                    top_left = (face_cords[0][0], face_cords[1][1])
                    bottom_right = (face_cords[1][0], face_cords[1][1] + 20)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, person_name, (face_cords[0][0] + 10, face_cords[1][1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, FONT_THICKNESS)
                    j += 1
            i += 1
            image = Image.fromarray(image)
            out_frames.append(image)
        self.video.release()
        return out_frames


input_video = 'short_video.mp4'
output_video = 'recognised_video.mp4'
face_recogniser = VideoFaceRecogniser()
face_locations, detected_faces = face_recogniser.detect_faces(input_video)
predictions = face_recogniser.recognise(detected_faces)
out_frames = face_recogniser.get_recognised_frames(input_video, predictions, face_locations)
face_recogniser.write_video(output_video, out_frames, 5)


