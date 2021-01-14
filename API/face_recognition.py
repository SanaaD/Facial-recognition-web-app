import numpy as np
import argparse
import cv2
import dlib
from os import listdir
from os.path import isdir
from scipy.spatial.distance import cosine
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
from keras.models import load_model
from tqdm import tqdm
import time


# Chemin vers le fichier Caffe 'deploy' prototxt
prototxt = 'models/deploy.prototxt.txt'

# Chemin vers le fichier Caffe du model pre-entrainé
# Model de DETECTION de visage entrainé sur une base de donnée de 140 000 personnes de race multiple
ssd_model = 'models/res10_300x300_ssd_iter_140000.caffemodel'

# Model d'extraction des trait du visages
predictor_model = "models/shape_predictor_68_face_landmarks.dat"

# Model de reconnaissance faciale de Google
facenet_model = 'models/facenet_keras.h5'

# seuil de confiance de detection de visage minimum
# si le seuil de confiance est suppèrieur à 55% on peut supposer qu'il s'agit d'un visage humain
min_confidence = 0.55


# Chargement des models nécessaire à la reconnaissance faciale
def load_models():

    print("Chargement du model de Reconnaissance Faciale...")

    pbar = tqdm(
        range(0, 3), desc="Chargement du model de détection de visages...")

    detector = cv2.dnn.readNetFromCaffe(prototxt, ssd_model)

    pbar.update(1)
    pbar.set_description(
        "Chargement du model d'extraction des traits du visages")

    predictor = dlib.shape_predictor(predictor_model)

    pbar.update(1)
    pbar.set_description("Chargement du model de reconnaissance facial")

    model = load_model(facenet_model, compile=False)

    pbar.update(1)
    pbar.set_description("Chargement des models terminé")
    pbar.close()

    return detector, predictor, model


# Chargement de tout les images pour une personnes donnée
def load_faces(directory, required_size=(160, 160)):
    faces = list()
    # list des images dans le dossier
    for filename in listdir(directory):
        # chemin de l'image
        path = directory + filename

        # chargement de l'image
        face = cv2.imread(path)

        # cRedimensionner l'image afin qu'elle soit accepté par le model facenet
        face_resized = cv2.resize(face, required_size)

        # sauvegarder l'image
        faces.append(face_resized)
    return faces


# Chargement de toutes les image
# pour chacune des peronnes de la base de données
def load_dataset(directory, model):
    X, y = list(), list()
    # list des dossier dans la base (1 par personnes)
    for subdir in listdir(directory):
        # chemin vers le dossier
        path = directory + subdir + '/'

        # si il s'agit d'un fichier passer au suivant
        if not isdir(path):
            continue

        # charger toute les images du dossier
        faces = load_faces(path)

        # créer les noms pour chacune des images
        labels = [subdir for _ in range(len(faces))]

        # résumé du nimbre d'image chargé pour la ième personne
        print('>loaded %d examples for class: %s' % (len(faces), subdir))

        # sauvegarder les images et les noms qui leur sont associés
        X.extend(faces)
        y.extend(labels)

    # si la liste n'est pas vide
    if len(X) > 0:
        X = get_embeddings(faces_pixels=asarray(X), model=model)

    return list(X), list(y)


def extract_faces(image, detector, required_size=(300, 300)):
    print("[INFO] extraction de visages...")

    # Récupération de la taille de l'image
    (s_height, s_width) = image.shape[:2]

    # Normaliser l'image (couleur vers gris) en utilisant (104.0, 177.0, 123.0) = > (rouge, vert, bleu)
    # les facteurs utilisés pour normaliser chaque couleur des pixels de l'image
    # Redimensionner l'image => 300 x 300px le format accepté par le model de detection de visage
    blob = cv2.dnn.blobFromImage(cv2.resize(image, required_size), 1.0,
                                 required_size, (104.0, 177.0, 123.0))

    # Initialisation du detecteur de visage avec le blob de l'image généré durant l'étape précedente
    detector.setInput(blob)

    # Detection des visages de l'image
    detections = detector.forward()

    # Tableau des viages (image Rogné pour garder uniquement les visages)
    detected_faces = []

    # Boucler sur les visages détecté
    for i in range(0, detections.shape[2]):

        # récupérer le taux de confiance pour le ième visage
        confidence = detections[0, 0, i, 2]

        # si le taux de confiance est suppèrieur au seuil
        if confidence > min_confidence:

            # calculer les coordonnée du visage sur l'image originel(non redimensionner)
            box = detections[0, 0, i, 3:7] * \
                np.array([s_width, s_height, s_width, s_height])
            (startX, startY, endX, endY) = box.astype("int")

            # Ajouter le visage au visage détecté
            detected_faces.append(dlib.rectangle(startX, startY, endX, endY))

    print("Fin extraction de visages")

    return detected_faces


def align_faces(faces, image, predictor, required_size=(160, 160)):
    # Récupération de la taille de l'image
    (s_height, s_width) = image.shape[:2]

    aligned_faces = []

    # Boucler sur les visages détecté
    for i, det in enumerate(faces):

        # Récupérer les traits du visage
        shape = predictor(image, det)

        # Récupérer les coordonnées des yeux grâce au traits du visages
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        # Récupérer la matice de rotation du visage grace à la position des yeux
        M = get_rotation_matrix(left_eye, right_eye)

        # Appliquer une rotation à l'image afin d'avoir le ième visage aligné
        rotated = cv2.warpAffine(
            image, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        # Rogner l'image afin de garder uniquement le ième visage
        cropped = crop_image(rotated, det)

        cropped = cv2.resize(cropped, required_size)

        # Ajouter le visage rogné à la liste
        aligned_faces.append(cropped)

    return asarray(aligned_faces)


# Vectorisation d'une liste de visages
def get_embeddings(faces_pixels, model):
    # Redimentionner les visage afin qu'ils soit acceptés par le modèle
    for i, face_pixels in enumerate(faces_pixels):
        faces_pixels[i] = faces_pixels[i].astype('float32')

    # Normalisation des pixel (entre 0 et 1) pour être utilisé avec facenet
    faces_pixels = faces_pixels / 255

    # si il s'agit d'une photo et non pas d'une liste de photo
    # ajouter une dimension au tableau de pixel afin qu'ils soit acceptés par le modèle
    # car le modèle prend plusieur visage en entré
    if(len(faces_pixels.shape) == 3):
        faces_pixels = expand_dims(faces_pixels, axis=0)

        # vectorisation du visage grâce au modele
    yhat = model.predict(faces_pixels)

    return yhat


# Calcule de la simlarité entre deux vecteurs de visages
# cosine similarity
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    return cosine(face_encodings, face_to_compare)
