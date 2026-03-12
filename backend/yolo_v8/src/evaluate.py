from ultralytics import YOLO

def evaluate():

    model = YOLO("../outputs/baseline_model/weights/best.pt")

    metrics = model.val()

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    evaluate()