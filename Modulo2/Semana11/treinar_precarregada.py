from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset.yaml",
    epochs=5,           
    imgsz=240,
    batch=2,
    project="meu_projeto",
    name="treino_finetuning" 
)