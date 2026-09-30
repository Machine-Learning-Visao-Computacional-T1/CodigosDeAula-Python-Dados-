# 🚀 API de Visão Computacional (YOLOv8 + FastAPI)

Este projeto recebe imagens via requisição HTTP e retorna as detecções de objetos em formato JSON.

## 🛠️ Como rodar localmente
1. Instale as dependências:
   `pip install -r requirements.txt`
2. Inicie o servidor:
   `uvicorn api_visao:app --reload`
3. Acesse a documentação em: `http://localhost:8000/docs`