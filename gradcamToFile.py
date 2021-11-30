import os

import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from main import *

device = 'cpu'

modelRestNet = models.resnet50(pretrained=True)
num_ftrs = modelRestNet.fc.in_features
modelRestNet.fc = nn.Linear(num_ftrs, 9)
modelRestNet.to(device)

modelVGG = models.vgg11(pretrained=True)
modelVGG.to(device)

modelRestNet.load_state_dict(torch.load("./model.pt", map_location=device))
modelVGG.load_state_dict(torch.load("./modelo_11ep_vgg_adagrad.pt", map_location=device))

modelList = [
    # model, target_layers, name
    (modelRestNet, [modelRestNet.layer4[-1]], "restnet"),
    (modelVGG, [modelVGG.features[-1]], "vgg")
]


def model_img_evaluate(img):
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    preprocess = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        norm,
    ])
    # Preprocess and display image
    image_tensor = preprocess(img)
    input_tensor = image_tensor.unsqueeze(0)  # single-image batch as wanted by model
    input_tensor = input_tensor.to(device)
    return input_tensor


BETTER = True
name = "dermatofibroma"
path = f"./moles/{name}.jpg"
if __name__ == '__main__':
    img = Image.open(path)
    input_tensor = model_img_evaluate(img)

    crop = cv2.resize(cv2.imread(path), (200, 200))
    savePath = f"./moles/{name}/"
    os.mkdir(savePath)
    cv2.imwrite(savePath + "original.jpg", crop)

    for t in modelList:
        model = t[0]
        target_layers = t[1]
        modelName = t[2]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True, aug_smooth=BETTER)
        grayscale_cam = grayscale_cam[0, :]
        original = np.transpose(np.squeeze(input_tensor).numpy(), (1, 2, 0))
        output = show_cam_on_image(original, grayscale_cam, use_rgb=True)
        cv2.imwrite(savePath + f"{modelName}.jpg", output)
