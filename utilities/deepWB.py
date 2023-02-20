"""
  Main function of inference phase for SWBNet

  Reference：
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""


import numpy as np
import torch
from torchvision import transforms
import utilities.utils as utls

from PIL import Image
import time


def deep_wb(image, net_awb=None,device='cuda'):
    image = Image.fromarray(image)
    image_resized = image.resize((656,656))
    # image_resized = image
    image = np.array(image)
    image_resized = np.array(image_resized)
    img = image_resized.transpose((2, 0, 1))
    img = img / 255
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    start_time = time.time()
    net_awb.eval()
    with torch.no_grad():

        output_awb = net_awb(img)



    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    output_awb = tf(torch.squeeze(output_awb.cpu()))
    output_awb = output_awb.squeeze().cpu().numpy()
    output_awb = output_awb.transpose((1, 2, 0))
    m_awb = utls.get_mapping_func(image_resized, output_awb)
    output_awb = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_awb))

    end_time = time.time()
    print(end_time-start_time)

    return output_awb