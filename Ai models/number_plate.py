from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data="BIKE NUMBER PLATES\data.yaml",epochs=50,imgsz=640,batch=8,workers=0,project='number_plate',name='number_plate_training',device=0)
model.export(format='onnx')
print('Training completed and model exported to ONNX format.')