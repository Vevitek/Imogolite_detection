import cv2
from skimage import img_as_ubyte, img_as_float
from skimage import io
import numpy as np
import matplotlib as plt
rho = 1
thresh_hough = 1
min_ll = 5
max_lg = 5
def Hough_detect(im1,im2):
    global im1g_normalized,im2g_normalized,rho,thresh_hough, min_ll, max_lg
    global im1_normalized
    im1_path= im1
    im1 = img_as_float(io.imread(im1))
    im1_normalized = (im1 - np.min(im1)) / (np.max(im1) - np.min(im1))
    im1g_normalized = (im1_normalized * 255).astype(np.uint8)

    im1_normalized_display = cv2.resize(im1_normalized, (614, 614))
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Im1", im1g)

    # im2 = img_as_float(io.imread(im2))
    im2_normalized = (im2 - np.min(im2)) / (np.max(im2) - np.min(im2))
    im2g_normalized = (im2_normalized * 255).astype(np.uint8)

    im2g_normalized_display = cv2.resize(im2g_normalized, (614, 614))
    # im2g_normalized = cv2.cvtColor(im2g_normalized, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLinesP(im2g_normalized, rho, np.pi / 180, thresh_hough, min_ll, max_lg)
    lines = np.squeeze(lines)

    cv2.imshow('Hough lines',im1_normalized_display)
    cv2.createTrackbar('Rho', 'Hough lines',rho, 10, rho_threshold_change)
    cv2.createTrackbar('Thresh_HLP', 'Hough lines', thresh_hough, 300, THLP_threshold_change)
    cv2.createTrackbar('Min line length', 'Hough lines', min_ll, 300, min_ll_threshold_change)
    cv2.createTrackbar('Max line gap', 'Hough lines', max_lg, 100, max_lg_threshold_change)

    while True:
        # Attendre que l'utilisateur appuie sur la touche 'q' pour quitter
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    lines = cv2.HoughLinesP(im2g_normalized, rho, np.pi / 180, thresh_hough, min_ll, max_lg)
    lines = np.squeeze(lines)

    if lines is not None:
        res = display_lines(im1g_normalized.copy(), lines)
        print("good")
    else:
        print("No lines detected.")

    cv2.imwrite("images\Hough_d.png",res)

def display_lines(image, lines):
    global im1g_normalized, im2g_normalized, rho, thresh_hough, min_ll, max_lg
    # image = cv2.imread(image)
    image_display = cv2.resize(image,(614,614))
    for line in lines:
        x1, y1,x2, y2 = line.tolist()
        col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.line(image,(x1,y1),(x2,y2),col,2)

    cv2.imshow('Line detection',image_display)

    return  image
def rho_threshold_change(value):
    global  im1g_normalized,im2g_normalized,rho,thresh_hough, min_ll, max_lg
    rho = value

    lines = cv2.HoughLinesP(im2g_normalized, rho, np.pi / 180, thresh_hough, min_ll, max_lg)
    lines = np.squeeze(lines)

    res = display_lines(im1g_normalized.copy(), lines)

    cv2.imshow('Line detection', res)
def THLP_threshold_change(value):
    global im1g_normalized, im2g_normalized, rho, thresh_hough, min_ll, max_lg
    thresh_hough = value

    lines = cv2.HoughLinesP(im2g_normalized, rho, np.pi / 180, thresh_hough, min_ll, max_lg)
    lines = np.squeeze(lines)

    res = display_lines(im1g_normalized.copy(), lines)

    cv2.imshow('Line detection', res)
def min_ll_threshold_change(value):
    global im1g_normalized, im2g_normalized, rho, thresh_hough, min_ll, max_lg
    min_ll = value

    lines = cv2.HoughLinesP(im2g_normalized, rho, np.pi / 180, thresh_hough, min_ll, max_lg)
    lines = np.squeeze(lines)

    res = display_lines(im1g_normalized.copy(), lines)

    cv2.imshow('Line detection', res)


def max_lg_threshold_change(value):
    global im1g_normalized, im2g_normalized, rho, thresh_hough, min_ll, max_lg
    max_lg = value

    lines = cv2.HoughLinesP(im2g_normalized, rho, np.pi / 180, thresh_hough, min_ll, max_lg)
    lines = np.squeeze(lines)

    res = display_lines(im1g_normalized.copy(), lines)

    cv2.imshow('Line detection', res)