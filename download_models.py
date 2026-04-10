from huggingface_hub import hf_hub_download
import os

os.makedirs('/app/models', exist_ok=True)

hf_hub_download('eerabhatt/ao-damage-models', 'parts_model.pth', local_dir='/app/models')
hf_hub_download('eerabhatt/ao-damage-models', 'damage_model.pth', local_dir='/app/models')
hf_hub_download('eerabhatt/ao-damage-models', 'best_car_damage_yolo.pt', local_dir='/app/models')

try:
    hf_hub_download('eerabhatt/ao-damage-models', 'severity_yolov8_cls.pt', local_dir='/app/models')
    print('Severity model downloaded')
except Exception as e:
    print(f'Severity model not yet available, using heuristic fallback: {e}')

print('Models downloaded successfully')
