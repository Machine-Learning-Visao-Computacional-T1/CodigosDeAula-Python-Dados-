from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import os

# ==========================================
# 1. MOLDES DE DADOS (PYDANTIC) PARA IMAGENS
# ==========================================
class Coordenadas(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class ObjetoDetectado(BaseModel):
    classe: str
    confianca_pct: float
    coordenadas: Coordenadas

class RespostaAPI(BaseModel):
    mensagem: str
    total_objetos: int
    resultados: list[ObjetoDetectado]

# ==========================================
# 2. INICIALIZAÇÃO DA API E DO MODELO
# ==========================================
app = FastAPI(title="API Full-Stack de Visão Computacional")

print("[SISTEMA] Carregando modelo YOLO...")
modelo = YOLO('yolov9m.pt')

# ==========================================
# 3. ROTA 1: PROCESSAR IMAGEM (Retorna JSON)
# ==========================================
@app.post("/detectar-imagem/", response_model=RespostaAPI)
async def analisar_imagem(arquivo: UploadFile = File(...)):
    
    if not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie uma imagem válida.")
    
    imagem_bytes = await arquivo.read()
    nparr = np.frombuffer(imagem_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resultados = modelo.predict(source=img_cv2, conf=0.25, verbose=False)
    resultado = resultados[0]

    deteccoes = []
    if len(resultado.boxes) > 0:
        for box in resultado.boxes:
            cls_id = int(box.cls[0].item())
            nome_classe = modelo.names[cls_id].upper()
            confianca = round(float(box.conf[0].item()) * 100, 2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            deteccoes.append(
                ObjetoDetectado(
                    classe=nome_classe,
                    confianca_pct=confianca,
                    coordenadas=Coordenadas(x_min=int(x1), y_min=int(y1), x_max=int(x2), y_max=int(y2))
                )
            )

    return RespostaAPI(
        mensagem="Análise de imagem concluída.",
        total_objetos=len(deteccoes),
        resultados=deteccoes
    )

# ==========================================
# 4. ROTA 2: PROCESSAR VÍDEO (Corrigida e Blindada)
# ==========================================
@app.post("/detectar-video/")
async def analisar_video(arquivo: UploadFile = File(...)):
    try:
        # Cria um diretório absoluto para evitar erros de caminho no Windows
        diretorio_base = os.path.abspath("temp_video")
        os.makedirs(diretorio_base, exist_ok=True)
        
        caminho_original = os.path.join(diretorio_base, arquivo.filename)
        
        # Salva o arquivo enviado no disco
        with open(caminho_original, "wb") as buffer:
            shutil.copyfileobj(arquivo.file, buffer)
        
        print(f"[API] Processando vídeo: {caminho_original}")
        
        # Executa a predição salvando no disco
        # Usamos save=True e indicamos a pasta de projeto
        resultados = modelo.predict(
            source=caminho_original, 
            save=True, 
            project=diretorio_base, 
            name="saida_processada", 
            exist_ok=True,
            conf=0.25,
            verbose=False
        )
        
        # O YOLO salva o vídeo processado dentro da pasta do projeto com o mesmo nome original
        nome_arquivo = os.path.basename(caminho_original)
        caminho_processado = os.path.join(diretorio_base, "saida_processada", nome_arquivo)
        
        # Fallback de extensão: no Windows o OpenCV/YOLO as vezes converte .mp4 para .avi
        if not os.path.exists(caminho_processado):
            nome_sem_ext, _ = os.path.splitext(nome_arquivo)
            caminho_processado_avi = os.path.join(diretorio_base, "saida_processada", f"{nome_sem_ext}.avi")
            if os.path.exists(caminho_processado_avi):
                caminho_processado = caminho_processado_avi

        if os.path.exists(caminho_processado):
            print(f"[API] Vídeo processado com sucesso em: {caminho_processado}")
            return FileResponse(
                caminho_processado, 
                media_type="video/mp4", 
                filename=f"analisado_{arquivo.filename}"
            )
        else:
            raise HTTPException(status_code=500, detail="O YOLO não gerou o arquivo de saída na pasta esperada.")

    except Exception as e:
        print(f"[ERRO NO VÍDEO] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar vídeo: {str(e)}")