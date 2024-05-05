from ultralytics import YOLO

# load a model
model = YOLO("yolov8n.pt") # load a pretrained model
# model = YOLO("./runs/detect/train/weights/best.pt") # load the old model

# train the model
if __name__ == '__main__':
    # results = model.train(data='C:/Users/Lachlan/Desktop/Cosc428/project/code/pyramid-traveler/data.yaml', epochs=500, imgsz=640)
    results = model.train(data='C:/Users/Lachlan/Desktop/Cosc428/project/code/rooms/data.yaml', epochs=500, imgsz=640)