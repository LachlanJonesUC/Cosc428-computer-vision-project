from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

# model = YOLO("yolov8n.pt") # load a pretrained model
model = YOLO("./runs/detect/train/weights/best.pt") # load the old model

if __name__ == '__main__':
    validation = model.val(data="./pyramid-traveler/data.yaml")