from ultralytics import YOLO

model = YOLO("yolo11n.yaml")

results = model.train(
    data="dataset.yaml", 
    epochs=20,           
    imgsz=520,           
    batch=2,
    project="meu_projeto",
    name="treino_do_zero",
    workers=0,
    degrees=25.0,   # Rotaciona a imagem entre -25 e +25 graus (frutas tortas na esteira)
    scale=0.5,      # Aplica Zoom In/Out (simula a câmera mais perto ou mais longe)
    fliplr=0.5,     # 50% de chance de espelhar horizontalmente (Esq/Dir)
    flipud=0.2,     # 20% de chance de espelhar verticalmente (Cima/Baixo)
    hsv_s=0.7,      # Altera a Saturação (simula câmeras de qualidade diferente)
    hsv_v=0.4,      # Altera o Brilho (simula iluminação ruim ou sombras na fábrica)
    mosaic=1.0,     # Ativa o Mosaico: Junta 4 imagens em 1 só (excelente para o YOLO aprender contexto)
    erasing=0.4     # Apaga pedaços aleatórios da imagem (força a IA a reconhecer a fruta mesmo escondida)    
)