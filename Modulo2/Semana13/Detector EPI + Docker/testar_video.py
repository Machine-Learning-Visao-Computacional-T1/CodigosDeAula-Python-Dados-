import cv2
from ultralytics import YOLO

# 1. Caminho absoluto do seu modelo treinado
caminho_modelo = r"F:\Senai\Detector EPI\dataset\runs\detect\modelo_epi\treino_yolo11\weights\best.pt"
model = YOLO(caminho_modelo)

# 2. Caminho exato do seu vídeo de teste (usando o 'r' na frente!)
fonte_video = r"F:\Senai\Detector EPI\dataset\test\videos\01 (2).mp4" 
cap = cv2.VideoCapture(fonte_video)

# Dicionário de cores (BGR) e textos
config_classes = {
    0: {"cor": (0, 255, 0), "texto": "Capacete"},
    1: {"cor": (255, 255, 0), "texto": "Camisa"},
    2: {"cor": (0, 255, 0), "texto": "Colete"},
    3: {"cor": (0, 255, 0), "texto": "Luva"},
    4: {"cor": (0, 0, 255), "texto": "SEM CAPACETE"},
    5: {"cor": (0, 0, 255), "texto": "SEM LUVA"}
}

print(f"Iniciando processamento do vídeo: {fonte_video}")
print("Pressione 'Q' na janela do vídeo para sair a qualquer momento.")

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        print("Fim do vídeo ou falha ao ler o arquivo.")
        break

    # 3. Faz a previsão no frame atual
    resultados = model.predict(frame, conf=0.45, verbose=False)[0]

    alerta_perigo = False

    # 4. Pinta os quadrados na imagem
    for box in resultados.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        if cls_id not in config_classes:
            continue
            
        cor = config_classes[cls_id]["cor"]
        texto = config_classes[cls_id]["texto"]
        
        # Se achar cabeça desprotegida (4) ou mão sem luva (5), ativa o alerta geral!
        if cls_id == 4 or cls_id == 5:
            alerta_perigo = True
        
        # Desenha a caixa individual do objeto
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)

    # 5. LÓGICA DO PAINEL GERAL (AVISO NA TELA)
    altura_frame, largura_frame = frame.shape[:2]
    
    # Desenha uma tarja preta no topo da tela
    cv2.rectangle(frame, (0, 0), (largura_frame, 50), (0, 0, 0), -1) 

    if alerta_perigo:
        # Tarja e Borda Vermelha
        cv2.putText(frame, "STATUS: PERIGO - TRABALHADOR SEM EPI!", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (largura_frame, altura_frame), (0, 0, 255), 10)
    else:
        if len(resultados.boxes) > 0:
            # Tarja Verde (Tudo OK)
            cv2.putText(frame, "STATUS: SEGURO - COM EPI", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            # Monitorando (Ninguém na tela)
            cv2.putText(frame, "STATUS: MONITORANDO...", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # 6. Exibe o vídeo redimensionado para não estourar a tela do seu PC (opcional, ajustando pra HD)
    frame_redimensionado = cv2.resize(frame, (1280, 720))
    cv2.imshow("Monitoramento de EPI ao Vivo", frame_redimensionado)
    
    # Sai se o usuário apertar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpa a memória ao terminar
cap.release()
cv2.destroyAllWindows()