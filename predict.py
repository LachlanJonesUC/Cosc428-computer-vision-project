from ultralytics import YOLO

model = YOLO('../runs/detect/train/weights/best.pt')

results = model('C:/Users/Lachlan/Desktop/Cosc428/project/code/holy.png')
results[0].save()

# model.predict('C:/Users/Lachlan/Desktop/Cosc428/project/code/pyramid-traveler/test/images/frame1068_png.rf.5cebff9fb1f68462749a3bfda00ec3ed.jpg', show=True)