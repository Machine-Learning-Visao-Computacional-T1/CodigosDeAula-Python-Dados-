from ultralytics import YOLO

caminho_melhor_modelo = r"C:\Users\welin\OneDrive\Área de Trabalho\projeto_frutas\runs\detect\meu_projeto\treino_dia_2\weights\best.pt"
model = YOLO(caminho_melhor_modelo)

imagem_nova = "imagens_teste/"

print(f"Analisando a imagem: {imagem_nova}...")


resultados = model.predict(source=imagem_nova, save=True, conf=0.1)


for r in resultados:
    print("\n--- Resultado da Detecção ---")
    print(f"Encontrado: {len(r.boxes)} objeto(s)")
    
    for box in r.boxes:
        classe_id = int(box.cls[0])           
        nome_classe = model.names[classe_id]  
        confianca = float(box.conf[0]) * 100  
        
        print(f"Objeto: {nome_classe} | Certeza: {confianca:.2f}%")

print("\nA imagem com o quadrado desenhado foi salva automaticamente na pasta: runs/detect/predict/")