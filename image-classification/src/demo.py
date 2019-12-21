import argparse

import numpy as np
from PIL import Image
import torch

from models import AlexNet
from train import CLASS_NAMES, num_classes

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--img_path',
                    default='./dog.jpg')
parser.add_argument('-w', '--weights_path',
                    default='../weights/30.pth')

args = parser.parse_args()


if __name__ == '__main__':
    # Prepare input image.
    img = Image.open(args.img_path).convert('RGB').resize((256, 256))
    img_array = np.asarray(img)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    c, h, w = img_tensor.size()
    x = img_tensor.view(1, c, h, w)

    # Load model and trained wights.
    model = AlexNet(num_classes=num_classes)
    params = torch.load(args.weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(params)
    model.eval()

    with torch.no_grad():
        output = model(x)
        _, y_pred = output.max(1)

        print(f'Prediction Result: {CLASS_NAMES[y_pred[0]]}')
