import cv2
from skimage import img_as_ubyte, img_as_float
from skimage import io

import numpy as np
import matplotlib as plt
# gray = None
lower_threshold = 50
upper_threshold = 255


def edge_detect(img):
    global gray, lower_threshold, upper_threshold
    percent_ori = 75

    # img = img_as_float(io.imread(img))
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    gray = (img_normalized * 255).astype(np.uint8)

    # cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
    # gray = cv2.resize(gray, (614,614))

    edges = cv2.Canny(gray,lower_threshold,upper_threshold)
    # cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    edges_display = cv2.resize(edges, (614,614))
    cv2.imshow("edges",edges_display)

    # Créer deux curseurs (trackbars) dans la fenêtre
    cv2.createTrackbar('Seuil inférieur', 'edges', lower_threshold, 255, on_lower_threshold_change)
    cv2.createTrackbar('Seuil supérieur', 'edges', upper_threshold, 255, on_upper_threshold_change)

    while True:
        # Attendre que l'utilisateur appuie sur la touche 'q' pour quitter
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Libérer les ressources et fermer les fenêtres OpenCV
    cv2.destroyAllWindows()

    edges = cv2.Canny(gray, lower_threshold, upper_threshold)

    cv2.imwrite('images\edges.png', edges)

    return edges

def on_lower_threshold_change(value):
    global gray
    global lower_threshold, upper_threshold

    # Ajustez le seuil inférieur en fonction de la valeur du curseur
    lower_threshold = value
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    edges_display = cv2.resize(edges,(614,614))

    # Affichez les contours détectés
    cv2.imshow('Contours détectés', edges_display)


def on_upper_threshold_change(value):
    global gray
    global lower_threshold, upper_threshold
    # Ajustez le seuil supérieur en fonction de la valeur du curseur
    upper_threshold = value
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    edges_display = cv2.resize(edges, (614, 614))

    # Affichez les contours détectés
    cv2.imshow('Contours détectés', edges_display)
