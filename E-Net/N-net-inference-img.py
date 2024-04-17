import cv2
import numpy as np
import e_net_architect
import torch
from matplotlib import pyplot as plt

model_path = 'ENET_second.pth'
enet_model = e_net_architect.ENet(2, 4)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
device = 'cpu'

enet_model.load_state_dict(torch.load(model_path))
enet_model.eval()

def process_and_visualize(input_image_path):
    input_image = cv2.imread(input_image_path)
    input_image = cv2.resize(input_image, (512, 256)) 
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = input_image[..., None]
    input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1) 

    with torch.no_grad():
        binary_logits, instance_logits = enet_model(input_tensor.unsqueeze(0))

    binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy()
    instance_seg = torch.argmax(instance_logits, dim=1).squeeze().numpy()

    plt.figure(figsize=(6, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(input_image.squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binary_seg, cmap='gray')
    plt.title('Binary Segmentation')
    plt.axis('off')

    plt.show()

input_img_path = 'laneImg.jpg'
process_and_visualize(input_img_path)


