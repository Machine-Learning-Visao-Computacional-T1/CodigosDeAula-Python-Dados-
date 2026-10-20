# 1. Puxa um Linux leve com Python
FROM python:3.10-slim

# 2. Instala dependências do sistema para visão computacional no Linux (Nomes atualizados)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 3. Define a pasta principal lá dentro do container
WORKDIR /app

# 4. Instala as bibliotecas Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia os seus arquivos (o código, a foto e o best.pt) para dentro do container
COPY . .

# 6. Manda executar o script quando o container ligar
CMD ["python", "testar_docker.py"]