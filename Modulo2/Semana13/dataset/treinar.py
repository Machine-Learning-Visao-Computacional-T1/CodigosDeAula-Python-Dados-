from ultralytics import YOLO

print("Iniciando o treinamento do Detector de EPI com YOLO11...")

# Baixa a versão mais moderna do YOLO
model = YOLO("yolo11n.pt") 

model.train(
    data="data.yaml",       
    epochs=30,              
    imgsz=640,              
    batch=4,                
    workers=0,             
    project="modelo_epi",   
    name="treino_yolo11"    
)

print("Treinamento finalizado! Seu modelo best.pt está salvo!")