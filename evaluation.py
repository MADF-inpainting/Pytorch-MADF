import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize
import numpy as np

def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    vgg_imgs = []
    with torch.no_grad():
        outputs = model(image.to(device), mask.to(device), gt.to(device))
    
    output = outputs[0][-1]
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output
    
    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))


    save_image(grid, filename)

