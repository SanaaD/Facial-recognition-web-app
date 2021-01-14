# -*- coding: latin-1 -*-
from flask import Flask, jsonify, request, make_response

import time
import numpy as np
import base64
import argparse
import cv2
import dlib
from utils import check_image, add_to_base
from tqdm import tqdm
from os import listdir, getcwd, mkdir
from os.path import isdir
from scipy.spatial.distance import cosine
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed

import face_recognition

ENCODING = 'utf-8'

# ------------------------------

app = Flask(__name__)

# ------------------------------

tic = time.time()

print("Chargement du model de Reconnaissance Faciale...")


# Chargement des modèles nécessaire à la reconnaissance faciale
detector, predictor, model = face_recognition.load_models()


# chargement de la base de données
database_path = getcwd() + "/database/"
print("database URI: ", database_path)
known_persons, labels = face_recognition.load_dataset(
    directory=database_path, model=model)


toc = time.time()

print("Model de Reconnaissance Faciale et base de données chargé en ",
      toc-tic, " seconds")


# ------------------------------
# Service API Interface


@app.route('/')
def index():
    return '<h1>API Launched</h1>'


@app.route('/identify', methods=['POST'])
def identify():

    # Récupérer la requette en json ------------------------------

    req = request.get_json()

    if "img" in list(req.keys()):

        img = req["img"]

        if not check_image(img):
            return jsonify({'success': False, 'errorMsg': 'une image au format string base64 est attendue'}), 205

        # Supprimer le "data:image/jpeg;base64" et garder uniquement l'image
        img = img.split(',')[1]

        # Decodage de l'image : format base64 vers une image
        im_bytes = base64.b64decode(img)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    else:
        return jsonify({'success': False, 'errorMsg': 'une image est attendue par l\'api'}), 205

    # Étape 1: Extraction des visages de l'image ------------------------------

    detected_faces = face_recognition.extract_faces(
        image=img, detector=detector)

    # Étape 2: Vérifier qu'il n'y est une personne sur l'image ------------------------------

    if len(detected_faces) < 1:
        return jsonify({'success': False, 'errorMsg': 'aucune personne présente sur l\'image'}), 500

    # Étape 3: Alignement des visages ------------------------------

    aligned_faces = face_recognition.align_faces(
        faces=detected_faces, image=img, predictor=predictor)

    # Étape 4: Encodage des visages pour reconnaissance faciale ------------------------------

    vectorized_faces = face_recognition.get_embeddings(
        faces_pixels=aligned_faces, model=model)

    # Étape 5: Identification ------------------------------

    result = []
    for unknown in vectorized_faces:

        best_cosine = 1.0
        best_label = 'inconnue'

        for i, known in enumerate(known_persons):

            face_dist = face_recognition.face_distance(unknown, known)

            if(best_cosine == 1.0 or best_cosine > face_dist):
                best_cosine = face_dist
                best_label = labels[i]

        if(best_cosine > 0.4):
            best_label = 'inconnue'

        result.append(best_label)

    # Étape 6: Ajout des noms sur l'image ------------------------------

    for i, face in enumerate(detected_faces):
        # On récupère les extrémité du visage pour tracer un rectangle autour
        startX, startY, endX, endY = face.left(), face.top(), face.right(), face.bottom()

        # On vérifie qu'on est pas au bord de l'image
        y = startY - 10 if startY - 10 > 10 else startY + 10

        # On ajoute le rectangle et le nom de la personne sur l'image
        cv2.rectangle(img, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(img, result[i], (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Encoder l'image en au format jpg
    retval, buffer = cv2.imencode('.jpg', img)

    # Encoder l'image en au format texte
    jpg_as_text = base64.b64encode(buffer).decode(ENCODING)

    return jsonify({"img": jpg_as_text}), 200


@app.route('/add', methods=['POST'])
def add_person():

    # Récupérer la requette en json ------------------------------

    req = request.get_json()

    if "img" in list(req.keys()):

        img = req["img"]

        if not check_image(img):
            return jsonify({'success': False, 'errorMsg': 'une image au format string base64 est attendue'}), 205

        # Supprimer le "data:image/jpeg;base64" et garder uniquement l'image
        img = img.split(',')[1]

        # Decodage de l'image : format base64 vers une image
        im_bytes = base64.b64decode(img)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    else:
        return jsonify({'success': False, 'errorMsg': 'une image est attendue par l\'api'}), 205

    if "name" in list(req.keys()):

        name = req["name"]

        # Vérifier si le nom est vide
        if not name.strip():
            return jsonify({'success': False, 'errorMsg': 'le nom de la personne est attendue'}), 205

        # Passer le nom en minuscule
        name = name.lower()

    else:
        return jsonify({'success': False, 'errorMsg': 'le nom de la personne est attendue par l\'api'}), 205

    # Étape 1: Extraction des visages de l'image ------------------------------

    detected_faces = face_recognition.extract_faces(
        image=img, detector=detector)

    # Étape 2: Vérifier qu'il n'ya qu'une personne sur l'image ------------------------------

    if len(detected_faces) > 1:
        return jsonify({'success': False, 'errorMsg': 'plusieurs personnes présentes sur l\'image'}), 500

    # Étape 3: Alignement du visage ------------------------------

    aligned_faces = face_recognition.align_faces(
        faces=detected_faces, image=img, predictor=predictor)

    # Étape 4: Ajout du visage à la base de donnée ------------------------------

    if not add_to_base(name, aligned_faces[0], database_path=database_path):
        return jsonify({'success': False, 'errorMsg': 'accès en écriture à la base de donnée impossible'}), 500

    # Étape 5: Encodage du visage ------------------------------

    vectorized_faces = face_recognition.get_embeddings(
        faces_pixels=aligned_faces, model=model)

    # Étape 6: Ajout du visage encodé à la liste des visages connus ------------------------------

    known_persons.append(vectorized_faces[0])
    labels.append(name)

    return jsonify({"success": True}), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5000,
        help='Port of serving api')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)
