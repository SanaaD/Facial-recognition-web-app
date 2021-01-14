import numpy as np
import cv2
import os


# API Récupéré d'internet ----------------------

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]


def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom


def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)


def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6


def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)


def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]


# Code Perso ----------------------

def check_image(image):
    valid_img = len(image) > 11 and image[0:11] == "data:image/"
    return valid_img


def add_to_base(name, image, database_path):

    path = database_path + name
    image_name = 'face_1.jpg'

    # Vérifier si l'utilisateur est présent dans la base
    # Si il ne l'est pas l'ajouter en créant un dossier à son nom
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Exhec lors de la création du dossier %s" % path)
            return False
        else:
            print("Création du dossier %s réussit" % path)

    # Récupérer la liste des image présente dans la base
    # pour la personne à ajouter (si il y en a)
    files = os.listdir(path)

    if len(files) > 0:

        # Si la liste n'est pas vide la trier par ordre décroissant
        files.sort(reverse=True)

        # Récupérer le nom de la dernière image ajouté
        # pour la personne en question
        last_file_name = files[0].split('.')[0]

        # Récupérer l'incrément de l'image et l'incrementer de 1
        increment = int(last_file_name.split('_')[1]) + 1

        # Mettre à jour le nom de la nouvelle image
        image_name = 'face_' + str(increment) + '.jpg'

    # Ajouter l'image dans la base de données
    cv2.imwrite(path + '/' + image_name, image)

    return True
