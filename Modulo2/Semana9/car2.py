import cv2
import random
from ultralytics import YOLO

def gerar_cores_dinamicas(nomes_classes):
    """Gera uma cor aleatória bonita para cada palavra nova que o usuário digitar"""
    cores = {}
    for nome in nomes_classes:
        # Gera cores vivas (evitando tons muito escuros para destacar no vídeo)
        cores[nome] = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    return cores

def executar_yolo_world():
    print("="*75)
    print("YOLO-WORLD: DETECÇÃO UNIVERSAL POR TEXTO LIVRE (ZERO-SHOT)")
    print("="*75)

    print("[SISTEMA] Baixando/Carregando motor YOLOv8s-World...")
    # Usamos o 's' (small) no World porque a arquitetura dele já é naturalmente pesada
    modelo = YOLO('yolov8s-world.pt') 

    # -------------------------------------------------------------
    # A MÁGICA DO YOLO WORLD: INPUT DE TEXTO LIVRE
    # -------------------------------------------------------------
    print("\n[IA GENERATIVA] O que você quer que a IA procure no vídeo?")
    print("Exemplo: mochila, pessoa, capacete branco, celular")
    texto_usuario = input("\nDigite os objetos separados por vírgula: ").strip()
    
    # Transforma o texto do usuário em uma lista de palavras (strip remove espaços extras)
    classes_alvo = [palavra.strip().lower() for palavra in texto_usuario.split(',')]
    
    # Injeta as palavras no "cérebro" da IA na mesma hora
    modelo.set_classes(classes_alvo)
    print(f"\n[SISTEMA] IA reconfigurada para detectar: {classes_alvo}")

    # Cria cores e contadores automáticos baseados no que o usuário digitou
    cores_classes = gerar_cores_dinamicas(classes_alvo)
    contagem_classes = {nome: 0 for nome in classes_alvo}

    fonte = input("\n[ 0 ] Webcam ou [ Nome do Arquivo.mp4 ]: ").strip()
    fonte = 0 if fonte == '0' else fonte.replace('"', '').replace("'", "")

    captura = cv2.VideoCapture(fonte)
    if not captura.isOpened():
        print("[ERRO] Falha ao abrir o vídeo.")
        return
    
    linha_y = 350 
    veiculos_contados = [] 

    while True:
        sucesso, frame = captura.read()
        if not sucesso: break

        frame = cv2.resize(frame, (1020, 600))

        # O track agora NÃO precisa mais do parâmetro 'classes=', pois o set_classes já limitou a rede!
        resultados = modelo.track(
            source=frame, 
            conf=0.10, # Confiança baixa porque a IA está deduzindo palavras novas
            iou=0.50,
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        resultado = resultados[0]

        cv2.line(frame, (0, linha_y), (1020, linha_y), (255, 255, 255), 1)

        if resultado.boxes.id is not None:
            caixas = resultado.boxes.xyxy.cpu().numpy().astype(int) 
            ids = resultado.boxes.id.cpu().numpy().astype(int)      
            classes = resultado.boxes.cls.cpu().numpy().astype(int) 
            confiancas = resultado.boxes.conf.cpu().numpy()

            for caixa, id_obj, cls_id, conf in zip(caixas, ids, classes, confiancas):
                x1, y1, x2, y2 = caixa
                centro_x, centro_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Pega o nome que o usuário digitou lá no começo
                nome_classe = modelo.names[cls_id].lower()
                porcentagem = int(conf * 100)
                cor = cores_classes.get(nome_classe, (255, 255, 255))

                texto_box = f"{nome_classe.upper()} #{id_obj} [{porcentagem}%]"

                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 250, y1), cor, -1) 
                
                cv2.putText(frame, texto_box, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.circle(frame, (centro_x, centro_y), 4, cor, -1)

                if (linha_y - 20) < centro_y < (linha_y + 20):
                    if id_obj not in veiculos_contados:
                        veiculos_contados.append(id_obj) 
                        contagem_classes[nome_classe] += 1
                        cv2.line(frame, (0, linha_y), (1020, linha_y), cor, 6)

        # -------------------------------------------------------------
        # HUD DINÂMICO (Se adapta ao número de palavras que o usuário digitou)
        # -------------------------------------------------------------
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (1020, 100), (0, 0, 0), -1) 
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "YOLO-WORLD: RADAR UNIVERSAL", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Desenha os contadores lado a lado dinamicamente
        posicao_x = 20
        for nome, qtd in contagem_classes.items():
            texto = f"{nome.upper()}: {qtd}"
            cv2.putText(frame, texto, (posicao_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cores_classes[nome], 2)
            posicao_x += 200 # Dá um espaçamento para desenhar a próxima palavra
            
        total = sum(contagem_classes.values())
        cv2.putText(frame, f"TOTAL: {total}", (820, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.imshow("Monitoramento YOLO-World", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    executar_yolo_world()