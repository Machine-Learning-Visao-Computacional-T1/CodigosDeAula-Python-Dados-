from ultralytics import YOLO


meu_modelo_atual = r"C:\Users\welin\OneDrive\Área de Trabalho\projeto_frutas\runs\detect\meu_projeto\treino_do_zero\weights\best.pt"
#se quer restaurar de onde parou, use o last.pt
#results = model.train(resume=True)

model = YOLO(meu_modelo_atual)

print("Iniciando uma nova rodada de treinamento")


model.train(
    data="dataset.yaml",       
    epochs=10,                 
    imgsz=320,
    batch=4,
    workers=0,
    project="meu_projeto",
    name="treino_dia_2"   
)

print("Treinamento extra concluído!")