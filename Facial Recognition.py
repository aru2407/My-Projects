# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:15:12 2020

@author: aruchakr
"""

from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
import os

path = "../Facial Recognition/images/Gopi/"
image = plt.imread(path + '1.jpg')

detector = MTCNN()

faces = detector.detect_faces(image)
for face in faces:
    print(face)

from matplotlib.patches import Rectangle

def highlight_faces(image_path, faces):
  # display image
    image = plt.imread(image_path)
    plt.imshow(image)

    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                          fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()
    
    
highlight_faces(path + '1.jpg', faces)


"""
SAME AS ABOVE IN A FUNCTION
Works on multiple faces. Change index in extracted faces.
"""
from numpy import asarray
from PIL import Image

def extract_face_from_image(image_path, required_size=(224, 224)):
  # load image and detect faces
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images

extracted_face = extract_face_from_image(path + '7.jpg')

# Display the first face from the extracted faces
plt.imshow(extracted_face[1])
plt.show()

"""COMPARE 2 FACES
"""

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')

    # perform prediction
    return model.predict(samples)

faces = extract_face_from_image(path+'1.jpg')
faces2 = extract_face_from_image(path+'2.jpg')
model_scores = get_model_scores(faces)
model_score2 = get_model_scores(faces2)

if cosine(model_scores, model_score2) <= 0.4:
  print("Faces Matched")