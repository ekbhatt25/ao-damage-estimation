from ultralytics import YOLO
import time

model = YOLO("../outputs/baseline_model/weights/best.pt")

start = time.time()

results = model.predict("../test_images/test.jpg")

latency = time.time() - start

print("Inference time:", latency)