import os
from ultralytics import YOLO

BASE_DIR = r"C:\Users\welin\OneDrive\Área de Trabalho\projeto_frutas"

pastas_imagens = [
    os.path.join(BASE_DIR, "images", "train"),
    os.path.join(BASE_DIR, "images", "val")
]

print("Carregando YOLO-World...")
model = YOLO("yolov8s-world.pt")

model.set_classes(["banana"])

confianca = 0.25

print("\nIniciando o Auto-Labeling...\n")

for pasta_img in pastas_imagens:
    if not os.path.exists(pasta_img):
        continue
        
    pasta_label = pasta_img.replace("images", "labels")
    os.makedirs(pasta_label, exist_ok=True)
    
    for nome_arquivo in os.listdir(pasta_img):
        if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            caminho_imagem = os.path.join(pasta_img, nome_arquivo)
            
            nome_txt = os.path.splitext(nome_arquivo)[0] + ".txt"
            caminho_txt = os.path.join(pasta_label, nome_txt)
            
            if os.path.exists(caminho_txt):
                continue
            
            resultados = model.predict(caminho_imagem, conf=confianca, verbose=False)
            
            with open(caminho_txt, 'w') as f:
                for box in resultados[0].boxes:
                    cls_id = int(box.cls[0])           
                    x_c, y_c, w, h = box.xywhn[0]      
                    
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            print(f"Marcado: {nome_arquivo} -> Encontrou {len(resultados[0].boxes)} objeto(s).")

print("\nAuto-Labeling concluído! Suas pastas /labels agora estão cheias de arquivos .txt!")