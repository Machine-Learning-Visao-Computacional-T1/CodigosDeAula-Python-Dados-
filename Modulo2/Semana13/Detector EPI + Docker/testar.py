import cv2
import os
import glob
from ultralytics import YOLO

# 1. Caminho absoluto do modelo treinado
caminho_modelo = r"F:\Senai\Detector EPI\best.pt"
model = YOLO(caminho_modelo)

# 2. Busca automaticamente a primeira imagem da sua pasta de testes
pasta_teste = r"F:\Senai\Detector EPI\dataset\test\images"
imagens = glob.glob(os.path.join(pasta_teste, "*.*"))

if not imagens:
    print("Nenhuma imagem encontrada na pasta de testes!")
    exit()

# Pega a primeira foto da lista (se quiser testar outras, mude o [0] para [1], [2], etc)
caminho_imagem = imagens[60]
print(f"Processando a imagem: {caminho_imagem}")

img = cv2.imread(caminho_imagem)

# Dicionário de cores (BGR) e textos
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

# Variável para controlar o alarme geral da foto
alerta_perigo = False

# 4. Pinta os quadrados individuais na imagem
for box in resultados.boxes:
    cls_id = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    if cls_id not in config_classes:
        continue
        
    cor = config_classes[cls_id]["cor"]
    texto = config_classes[cls_id]["texto"]
    
    # Checa se é a classe da cabeça sem capacete (4) ou mão sem luva (5)
    if cls_id == 4 or cls_id == 5:
        alerta_perigo = True
    
    # Desenha a caixa individual
    cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)
    cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2, cv2.LINE_AA)

# 5. LÓGICA DO PAINEL GERAL (A Tarja no Topo)
altura_img, largura_img = img.shape[:2]

# Tarja preta no topo (com 60 pixels de altura)
cv2.rectangle(img, (0, 0), (largura_img, 60), (0, 0, 0), -1) 

if alerta_perigo:
    # Escreve o texto em vermelho
    cv2.putText(img, "STATUS: PERIGO - TRABALHADOR SEM EPI!", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    # Borda grossa vermelha ao redor de toda a imagem para destacar o perigo
    cv2.rectangle(img, (0, 0), (largura_img, altura_img), (0, 0, 255), 15)
else:
    # Se achou itens, mas nenhum é de perigo
    if len(resultados.boxes) > 0:
        cv2.putText(img, "STATUS: SEGURO - COM EPI", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        # Borda grossa verde ao redor de toda a imagem
        cv2.rectangle(img, (0, 0), (largura_img, altura_img), (0, 255, 0), 15)
    else:
        cv2.putText(img, "STATUS: NENHUMA DETECCAO", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# 6. Salva a foto processada
nome_saida = "resultado_epi.jpg"
cv2.imwrite(nome_saida, img)
print(f"Pronto! Imagem processada e salva como '{nome_saida}'")