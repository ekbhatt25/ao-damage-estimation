import os
from ultralytics import YOLO
import torch

# train
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(base_dir, '..', 'configs', 'dataset.yaml'))
    output_dir = os.path.abspath(os.path.join(base_dir, '..', 'outputs', 'models'))

    if not os.path.exists(config_path):
        print(f"Error: Could not find configuration file at {config_path}")
        return

    # GPU
    if torch.cuda.is_available():
        device = 0 
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    # load pre-trained model
    model = YOLO('yolov8n.pt')

    # train
    model.train(
        data=config_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project=output_dir,
        name='baseline_run',
        device=device,
        exist_ok=True
    )

if __name__ == "__main__":
    train_model()



