import torch

print("="*50)
print("DIAGNÓSTICO DE HARDWARE - INTELIGÊNCIA ARTIFICIAL")
print("="*50)

suporte_cuda = torch.cuda.is_available()
print(f"CUDA Disponível e Ativo? {suporte_cuda}")

if suporte_cuda:
    print(f"Placa de Vídeo Detectada: {torch.cuda.get_device_name(0)}")
    print("Tudo pronto! Seu YOLO agora vai rodar na placa de vídeo.")
else:
    print("[ERRO] O Python ainda não consegue ver a GPU.")