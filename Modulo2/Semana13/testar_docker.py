import cv2
import os
from ultralytics import YOLO

print("Iniciando a IA dentro do Docker...")

# 1. Caminho relativo (o arquivo best.pt está na mesma pasta agora)
model = YOLO("best.pt")

# 2. Lê a foto que mapeamos
caminho_imagem = "hardhat_40_jpeg_jpg.rf.aaec14efd86bd2d1673bbaabe2e56dd8.jpg"

if not os.path.exists(caminho_imagem):
    print(f"Erro: A foto '{caminho_imagem}' não foi encontrada na pasta!")
    exit()

print("Processando a imagem...")
img = cv2.imread(caminho_imagem)

# Dicionário de regras
config_classes = {
    0: {"cor": (0, 255, 0), "texto": "Capacete"},
    1: {"cor": (255, 255, 0), "texto": "Camisa"},
    2: {"cor": (0, 255, 0), "texto": "Colete"},
    3: {"cor": (0, 255, 0), "texto": "Luva"},
    4: {"cor": (0, 0, 255), "texto": "SEM CAPACETE"},
    5: {"cor": (0, 0, 255), "texto": "SEM LUVA"}
}

# 3. Faz a previsão
resultados = model.predict(img, conf=0.45, verbose=False)[0]
alerta_perigo = False

for box in resultados.boxes:
    cls_id = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    if cls_id not in config_classes:
        continue
        
    cor = config_classes[cls_id]["cor"]
    texto = config_classes[cls_id]["texto"]
    
    if cls_id == 4 or cls_id == 5:
        alerta_perigo = True
    
    cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)
    cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2, cv2.LINE_AA)

# 4. Tarja superior
altura_img, largura_img = img.shape[:2]
cv2.rectangle(img, (0, 0), (largura_img, 60), (0, 0, 0), -1) 

if alerta_perigo:
    cv2.putText(img, "STATUS: PERIGO - TRABALHADOR SEM EPI!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (largura_img, altura_img), (0, 0, 255), 15)
else:
    if len(resultados.boxes) > 0:
        cv2.putText(img, "STATUS: SEGURO - COM EPI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (largura_img, altura_img), (0, 255, 0), 15)
    else:
        cv2.putText(img, "STATUS: NENHUMA DETECCAO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# 5. Salva a foto processada
nome_saida = "resultado_docker.jpg"
cv2.imwrite(nome_saida, img)
print(f"Sucesso! Imagem salva como '{nome_saida}'")