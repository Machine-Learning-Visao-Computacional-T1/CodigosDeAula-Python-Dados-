import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

# Configuração do visual do site
st.set_page_config(page_title="AI Vision Analytics", layout="wide")
st.title("👁️ AI Vision Analytics - SENAI")
st.markdown("Plataforma de processamento visual. Escolha o tipo de mídia abaixo:")

# Endereços da nossa API
URL_API_IMAGEM = "http://127.0.0.1:8000/detectar-imagem/"
URL_API_VIDEO = "http://127.0.0.1:8000/detectar-video/"

# Cria duas abas bonitas no site
aba_imagem, aba_video = st.tabs(["📸 Analisar Imagem", "🎥 Analisar Vídeo"])

# ==========================================
# ABA 1: FRONT-END DA IMAGEM
# ==========================================
with aba_imagem:
    st.header("Análise de Imagens Estáticas")
    arquivo_img = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"], key="img")

    if arquivo_img is not None:
        imagem = Image.open(arquivo_img)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(imagem, caption="Imagem Original", use_container_width=True)
            
        if st.button("🚀 Iniciar Análise de Imagem", use_container_width=True):
            with st.spinner('Acessando o servidor de Inteligência Artificial...'):
                
                # CORREÇÃO: Se a imagem tiver canal de transparência (RGBA/PNG), converte para RGB puro
                if imagem.mode in ("RGBA", "P"):
                    imagem = imagem.convert("RGB")
                
                # Prepara e envia para a API
                img_byte_arr = io.BytesIO()
                imagem.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                resposta = requests.post(URL_API_IMAGEM, files={"arquivo": ("foto.jpg", img_bytes, "image/jpeg")})
                
                if resposta.status_code == 200:
                    dados = resposta.json()
                    
                    # Desenha as Bounding Boxes na foto
                    img_desenhada = imagem.copy()
                    draw = ImageDraw.Draw(img_desenhada)
                    
                    for obj in dados['resultados']:
                        c = obj['coordenadas']
                        # Desenha a caixa
                        draw.rectangle([c['x_min'], c['y_min'], c['x_max'], c['y_max']], outline="lime", width=4)
                        # Desenha o texto do objeto
                        draw.text((c['x_min'], c['y_min'] - 10), f"{obj['classe']} {obj['confianca_pct']}%", fill="lime")
                    
                    with col2:
                        st.image(img_desenhada, caption=f"Achamos {dados['total_objetos']} objetos!", use_container_width=True)
                    
                    st.success("JSON retornado pela API:")
                    st.json(dados)
                else:
                    st.error("Erro no servidor da API.")

# ==========================================
# ABA 2: FRONT-END DO VÍDEO
# ==========================================
with aba_video:
    st.header("Análise de Vídeos em Movimento")
    st.warning("Atenção: O processamento de vídeo exige muito hardware. Vídeos grandes podem demorar alguns minutos.")
    
    arquivo_vid = st.file_uploader("Selecione um vídeo pequeno", type=["mp4", "avi", "mov"], key="vid")
    
    if arquivo_vid is not None:
        if st.button("🎬 Iniciar Análise de Vídeo", use_container_width=True):
            with st.spinner('A API está analisando quadro a quadro (Isso pode demorar)...'):
                
                # Prepara e envia o vídeo para a API
                resposta = requests.post(URL_API_VIDEO, files={"arquivo": (arquivo_vid.name, arquivo_vid.getvalue(), "video/mp4")})
                
                if resposta.status_code == 200:
                    st.success("Vídeo processado com sucesso!")
                    
                    # A API nos devolveu o arquivo de vídeo pronto. Vamos liberar para o usuário baixar!
                    st.download_button(
                        label="📥 Baixar Vídeo Processado",
                        data=resposta.content,
                        file_name=f"analisado_{arquivo_vid.name}.avi",
                        mime="video/x-msvideo"
                    )
                else:
                    st.error("Falha ao processar o vídeo na API.")