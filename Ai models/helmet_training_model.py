from ultralytics import YOLO
model = YOLO('yolov8n.pt')
	# Use local dataset YAML that points to workspace folders
model.train(
		data='archive/data/local_data.yaml',
		epochs=50,
		imgsz=640,
		batch=8,
		workers=0,
		project='helmet_project',
		name='helmet_training',
		device=0
	)
model.export(format='onnx')
print('Training completed and model exported to ONNX format.')


