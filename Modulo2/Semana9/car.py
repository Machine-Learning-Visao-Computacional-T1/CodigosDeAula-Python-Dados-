import cv2
import numpy as np
from ultralytics import YOLO

def obter_ids_veiculos_oiv7(modelo):
    alvos = ['car', 'motorcycle', 'bus', 'truck', 'van', 'taxi', 'ambulance', 'limousine']
    ids_encontrados = []
    print("\n[SISTEMA] Mapeando classes de veículos no dataset OIV7...")
    for id_classe, nome_classe in modelo.names.items():
        if nome_classe.lower() in alvos:
            ids_encontrados.append(id_classe)
    return ids_encontrados

# Dicionário de cores bonitas (BGR) para as caixas
CORES_CLASSES = {
    'car': (255, 144, 30),        # Azul Ciano
    'motorcycle': (0, 165, 255),  # Laranja
    'bus': (0, 255, 0),           # Verde
    'truck': (200, 0, 200),       # Roxo
    'van': (255, 255, 0),         # Ciano Claro
    'taxi': (0, 215, 255),        # Amarelo/Dourado
    'ambulance': (0, 0, 255)      # Vermelho
}

def executar_radar_premium():
    print("="*75)
    print("SMART CITY: RADAR PREMIUM COM GPU (OIV7)")
    print("="*75)

    modelo = YOLO('yolo26m.pt') 
    classes_veiculos = obter_ids_veiculos_oiv7(modelo)

    fonte = input("\n[ 0 ] Webcam ou [ Nome do Arquivo.mp4 ]: ").strip()
    fonte = 0 if fonte == '0' else fonte.replace('"', '').replace("'", "")

    captura = cv2.VideoCapture(fonte)
    if not captura.isOpened():
        print("[ERRO] Falha ao abrir o vídeo.")
        return
    
    linha_y = 350 
    veiculos_contados = [] 
    contagem_classes = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'van': 0, 'outros': 0}

    while True:
        sucesso, frame = captura.read()
        if not sucesso: break

        frame = cv2.resize(frame, (1020, 600))

        resultados = modelo.track(
            source=frame, 
            classes=classes_veiculos, 
            conf=0.15, 
            iou=0.50,
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        resultado = resultados[0]

        cv2.line(frame, (0, linha_y), (1020, linha_y), (255, 255, 255), 1)

        if resultado.boxes.id is not None:
            # EXTRAÇÃO DE DADOS: Adicionamos a confiança (conf) aqui
            caixas = resultado.boxes.xyxy.cpu().numpy().astype(int) 
            ids = resultado.boxes.id.cpu().numpy().astype(int)      
            classes = resultado.boxes.cls.cpu().numpy().astype(int) 
            confiancas = resultado.boxes.conf.cpu().numpy() # <-- Puxa a precisão decimal (ex: 0.85)

            # O zip agora une 4 informações ao mesmo tempo
            for caixa, id_obj, cls_id, conf in zip(caixas, ids, classes, confiancas):
                x1, y1, x2, y2 = caixa
                centro_x, centro_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                nome_classe = modelo.names[cls_id].lower()

                # Matemática simples: 0.857 * 100 = 85.7 -> convertendo pra inteiro = 85
                porcentagem = int(conf * 100)

                cor = CORES_CLASSES.get(nome_classe, (255, 255, 255))

                # Design da Bounding Box com o novo texto
                texto_box = f"{nome_classe.upper()} #{id_obj} [{porcentagem}%]"

                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                # Aumentei a largura do fundo de (x1 + 150) para (x1 + 210) para caber a porcentagem
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 210, y1), cor, -1) 
                
                cv2.putText(frame, texto_box, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                cv2.circle(frame, (centro_x, centro_y), 4, cor, -1)

                # Lógica de Contagem
                if (linha_y - 20) < centro_y < (linha_y + 20):
                    if id_obj not in veiculos_contados:
                        veiculos_contados.append(id_obj) 
                        if nome_classe in contagem_classes:
                            contagem_classes[nome_classe] += 1
                        else:
                            contagem_classes['outros'] += 1
                        cv2.line(frame, (0, linha_y), (1020, linha_y), cor, 6)

        # -------------------------------------------------------------
        # HUD TRANSPARENTE
        # -------------------------------------------------------------
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (1020, 100), (0, 0, 0), -1) 
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "RADAR SMART CITY", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        texto_carros = f"Carros: {contagem_classes['car']}"
        texto_motos = f"Motos: {contagem_classes['motorcycle']}"
        texto_pesados = f"Onibus/Caminhao: {contagem_classes['bus'] + contagem_classes['truck']}"
        
        cv2.putText(frame, texto_carros, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORES_CLASSES['car'], 2)
        cv2.putText(frame, texto_motos, (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORES_CLASSES['motorcycle'], 2)
        cv2.putText(frame, texto_pesados, (450, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORES_CLASSES['truck'], 2)
        
        total = sum(contagem_classes.values())
        cv2.putText(frame, f"TOTAL: {total}", (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.imshow("Sistema de Monitoramento V2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    executar_radar_premium()