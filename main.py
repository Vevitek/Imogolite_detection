from Denoise import *
from Edge_detection import *
from DigitalSreeni import *
from Hough_detection import *

img = r"E:\Mails\Imogolites\Second_test_PS.png"
# denoise_img = r"C:\Users\Zakar\PycharmProjects\Hough_Imo\venv\images\NLM.jpg"

denoise_img = Denoise(img)
edges = edge_detect(denoise_img)

Hough_detect(img, edges)




