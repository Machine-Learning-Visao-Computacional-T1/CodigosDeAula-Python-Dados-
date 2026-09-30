from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np

# ==============================================================================
# BLOCO 1: PYDANTIC (OS "MOLDES" DA NOSSA ENTREGA)
# Explicação: Aqui nós definimos o formato exato do texto (JSON) que a API vai 
# devolver. Isso é padrão ouro na indústria, pois os desenvolvedores Front-End
# que vão usar nossa API precisam saber exatamente o formato da resposta.
# ==============================================================================
class Coordenadas(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class ObjetoDetectado(BaseModel):
    classe: str
    confianca_pct: float
    coordenadas: Coordenadas  # Usa o molde de cima para as 4 posições

class RespostaAPI(BaseModel):
    mensagem: str
    total_objetos: int
    resultados: list[ObjetoDetectado] # Uma lista contendo vários objetos detectados

# ==============================================================================
# BLOCO 2: INICIALIZAÇÃO DOS SERVIDORES
# ==============================================================================
app = FastAPI(
    title="API Detector", 
    description="API para Detecção de Objetos usando YOLO26.",
    version="1.0.0"
)

print("[SISTEMA] Carregando o motor YOLO na memória...")
modelo = YOLO('yolo26l.pt')

# ==============================================================================
# BLOCO 3: ROTAS E PROCESSAMENTO DA IMAGEM
# ==============================================================================
@app.get("/")
async def status_servidor():
    """Rota simples apenas para testar se a API está online."""
    return {"status": "Online", "mensagem": "Motor de IA pronto para receber imagens."}

# O response_model=RespostaAPI diz ao FastAPI: "Sempre devolva os dados usando o Molde que criamos!"
@app.post("/detectar/", response_model=RespostaAPI)
async def detectar_objetos(arquivo: UploadFile = File(...)):
    """Recebe uma foto, traduz para o YOLO, analisa e devolve as coordenadas."""
    
    # ---------------------------------------------------------
    # PASSO A: SEGURANÇA E LEITURA DE BYTES DA INTERNET
    # ---------------------------------------------------------
    # Se o usuário tentar mandar um PDF ou arquivo de texto, a API bloqueia.
    if not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Você deve enviar uma imagem válida.")
    
    try:
        # A internet envia os arquivos em formato bruto (bytes). Precisamos ler isso.
        imagem_bytes = await arquivo.read()
        
        # ---------------------------------------------------------
        # PASSO B: TRADUÇÃO PARA O OPENCV (O cérebro visual)
        # ---------------------------------------------------------
        # O YOLO e o OpenCV não entendem "bytes da internet". Eles entendem MATRIZES matemáticas.
        # np.frombuffer: Converte os bytes puros em um vetor de números (NumPy)
        nparr = np.frombuffer(imagem_bytes, np.uint8)
        
        # cv2.imdecode: Transforma esse vetor na foto colorida estruturada que o YOLO exige
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
         
        # ---------------------------------------------------------
        # PASSO C: INFERÊNCIA DA REDE NEURAL
        # ---------------------------------------------------------
        # A placa de vídeo processa a foto instantaneamente
        resultados = modelo.predict(source=img_cv2, conf=0.40, verbose=False)
        resultado = resultados[0]

        # ---------------------------------------------------------
        # PASSO D: EXTRAÇÃO DOS DADOS PARA O MOLDE
        # ---------------------------------------------------------
        deteccoes = [] # Lista vazia para guardar o que acharmos
        
        if len(resultado.boxes) > 0:
            for box in resultado.boxes:
                # 1. Puxando as informações brutas do Tensor do YOLO
                cls_id = int(box.cls[0].item())
                nome_classe = modelo.names[cls_id].upper()
                confianca = round(float(box.conf[0].item()) * 100, 2)
                
                # Puxa as 4 coordenadas do retângulo (Bounding Box)
                x1, y1, x2, y2 = box.xyxy[0].tolist() 
                
                # 2. Encaixando os dados no nosso Molde Pydantic (ObjetoDetectado)
                deteccoes.append(
                    ObjetoDetectado(
                        classe=nome_classe,
                        confianca_pct=confianca,
                        coordenadas=Coordenadas(
                            x_min=int(x1), y_min=int(y1), 
                            x_max=int(x2), y_max=int(y2)
                        )
                    )
                )

        # ---------------------------------------------------------
        # PASSO E: A RESPOSTA FINAL AO USUÁRIO
        # ---------------------------------------------------------
        return RespostaAPI(
            mensagem="Análise de imagem concluída com sucesso.",
            total_objetos=len(deteccoes),
            resultados=deteccoes
        )
        
    except Exception as e:
        # Se ocorrer qualquer falha técnica (ex: foto corrompida), evita que o servidor "caia"
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {str(e)}")