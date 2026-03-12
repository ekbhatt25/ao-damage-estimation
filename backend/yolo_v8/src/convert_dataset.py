import os
import json
import cv2
import shutil
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/car_damage_dataset/Car damages dataset/File1"
IMG_PATH = os.path.join(DATA_PATH, "img")
ANN_PATH = os.path.join(DATA_PATH, "ann")

OUTPUT_PATH = "../data/yolo_dataset"

os.makedirs(f"{OUTPUT_PATH}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels/val", exist_ok=True)

images = [f for f in os.listdir(IMG_PATH) if f.endswith(".jpg")]

train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    return x_min, y_min, x_max - x_min, y_max - y_min


def convert_bbox(size, bbox):
    w, h = size
    x, y, bw, bh = bbox

    x_center = (x + bw / 2) / w
    y_center = (y + bh / 2) / h
    bw /= w
    bh /= h

    return x_center, y_center, bw, bh


for split, img_list in [("train", train_imgs), ("val", val_imgs)]:

    for img_name in img_list:

        img_path = os.path.join(IMG_PATH, img_name)
        ann_path = os.path.join(ANN_PATH, img_name.replace(".jpg", ".json"))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        shutil.copy(img_path, f"{OUTPUT_PATH}/images/{split}/{img_name}")

        label_path = f"{OUTPUT_PATH}/labels/{split}/{img_name.replace('.jpg','.txt')}"

        with open(label_path, "w") as label_file:

            if os.path.exists(ann_path):

                with open(ann_path) as f:
                    data = json.load(f)

                for obj in data["objects"]:

                    polygon = obj["polygon"]

                    bbox = polygon_to_bbox(polygon)

                    x, y, bw, bh = convert_bbox((w, h), bbox)

                    class_id = 0

                    label_file.write(f"{class_id} {x} {y} {bw} {bh}\n")

print("Dataset conversion complete.")
