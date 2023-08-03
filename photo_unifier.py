import mtcnn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import os
FACES_DIR = "faces"
NORMALISED_FACES_DIR = "faces_base"

face_detector = mtcnn.MTCNN()

for name in os.listdir(FACES_DIR):
    for filename in os.listdir(f'{FACES_DIR}/{name}'):
        print(f'{FACES_DIR}/{name}/{filename}')
        image = plt.imread(f'{FACES_DIR}/{name}/{filename}')
        face_location = face_detector.detect_faces(image)
        x1, y1, width, height = face_location[0]['box']
        x2, y2 = x1 + width, y1 + height
        located_face = image[y1:y2, x1:x2]
        face_image = Image.fromarray(located_face).resize((224, 224))
        face_image.save(f'{NORMALISED_FACES_DIR}/{name}/{filename}')

