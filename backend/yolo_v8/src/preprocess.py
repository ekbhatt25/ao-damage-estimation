import cv2
import os

def blur_score(image_path):
    img = cv2.imread(image_path)
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return score

def filter_blurry_images(image_folder, threshold=100):
    removed = []

    for img in os.listdir(image_folder):
        path = os.path.join(image_folder, img)

        if blur_score(path) < threshold:
            removed.append(path)

    return removed