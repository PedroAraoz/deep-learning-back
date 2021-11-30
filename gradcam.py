
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from main import *

device = 'cpu'

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)
model.to(device)

lr = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=lr)

model.load_state_dict(torch.load("./model.pt", map_location=device))
target_layers = [model.layer4[-1]]


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


def waitKey():
    if cv2.waitKey() == ord('q'):
        exit()


windowName = 'GradCAM'
cv2.namedWindow(windowName)

BETTER = False
path = ""
if __name__ == '__main__':
    img = Image.open(path)
    input_tensor = model_img_evaluate(img)
    # Create an input tensor image for your model..

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    # target_category = 6

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True, aug_smooth=BETTER)

    grayscale_cam = grayscale_cam[0, :]

    original = np.transpose(np.squeeze(input_tensor).numpy(), (1, 2, 0))

    output = show_cam_on_image(original, grayscale_cam, use_rgb=True)

    while True:
        crop = cv2.resize(cv2.imread(path), output.shape[0:2])
        cv2.imshow(windowName, crop)
        waitKey()
        cv2.imshow(windowName, output)
        waitKey()
