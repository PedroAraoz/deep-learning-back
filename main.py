import datetime
import io

import uvicorn
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim

app = FastAPI()

device = 'cpu'

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)
model.to(device)

lr = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=lr)

labels = ['actinic keratosis',  # doctor
          'basal cell carcinoma',  # doctor
          'dermatofibroma',  # no
          'melanoma',  # doctor
          'nevus',  # no
          'pigmented benign keratosis',  # no
          'seborrheic keratosis',  # no
          'squamous cell carcinoma',  # doctor
          'vascular lesion']  # no

doctorLables = [
    'actinic keratosis', 'basal cell carcinoma', 'melanoma', 'squamous cell carcinoma'
]


@app.on_event("startup")
def startup_event():
    start = datetime.datetime.now()
    print("Loading model...")
    global model
    model.load_state_dict(torch.load("./model.pt", map_location=device))
    model.eval()
    ms = (datetime.datetime.now() - start).microseconds / 100
    print(f"Model loaded in {ms} ms! :D")


@app.post("/analyze")
async def analyzeImage(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = model_img_evaluate(img)
    return {"predictions": prediction}


@app.post("/simpleAnalyze")
async def simpleAnalyzeImage(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = model_img_evaluate(img)
    return shouldGoToDoctor(prediction)


@app.post("/completeAnalyze")
async def simpleAnalyzeImage(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = model_img_evaluate(img)
    print(prediction[0:2])
    return {
        "doctor": shouldGoToDoctor(prediction),
        "predictions": prediction
    }


def model_img_evaluate(img):
    # Preprocess and display image
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    inv_norm = transforms.Normalize((-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5))
    preprocess = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        norm,
    ])
    image_tensor = preprocess(img)
    input_tensor = image_tensor.unsqueeze(0)  # single-image batch as wanted by model
    input_tensor = input_tensor.to(device)

    # Single prediction call
    outputs = model(input_tensor)
    outputs = torch.topk(outputs, 9)
    indices = outputs.indices[0]
    values = outputs.values[0]
    res = []
    for i in range(len(indices)):
        res.append({labels[indices[i].item()]: values[i].item()})
    return res


def shouldGoToDoctor(prediction):
    return len(list(filter(lambda x: list(x.keys())[0] in doctorLables, prediction[0:2]))) > 0


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
