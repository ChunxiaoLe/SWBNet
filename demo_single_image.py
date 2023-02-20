"""
 Demo single image
"""

import argparse
import logging
import os
import torch
from PIL import Image
from arch import deep_wb_model_msa
from utilities.deepWB import deep_wb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



def get_args():
    parser = argparse.ArgumentParser(description='Changing WB of an input image.')
    parser.add_argument('--model_dir', '-m', default='./models',
                        help="Specify the directory of the trained model.", dest='model_dir')
    parser.add_argument('--input', '-i', help='Input image filename', dest='input',
                        default='./example_images/1127_D.JPG')
    parser.add_argument('--output_dir', '-o', default='./result_images',
                        help='Directory to save the output images', dest='out_dir')
    parser.add_argument('--task', '-t', default='all',
                        help="Specify the required task: 'AWB', 'editing', or 'all'.", dest='task')

    parser.add_argument('--mxsize', '-S', default=656, type=int,
                        help="Max dim of input image to the network, the output will be saved in its original res.",
                        dest='S')
    parser.add_argument('--show', '-v', action='store_true', default=True,
                        help="Visualize the input and output images",
                        dest='show')
    parser.add_argument('--save', '-s', action='store_true',
                        help="Save the output images",
                        default=True, dest='save')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    fn = args.input
    out_dir = args.out_dir
    S = args.S
    tosave = args.save

    "loading model"
    net_awb = deep_wb_model_msa.deepWBNet()
    net_awb.load_state_dict(
        torch.load(os.path.join(args.model_dir,
                                'net_ctif.pth')
                   , map_location='cuda:0'))

    net_awb.to(device=device)
    net_awb.eval()

    "input image"
    logging.info("Processing image {} ...".format(fn))
    img = Image.open(fn)
    plt.imshow(img)
    plt.show()

    "white-balanced image"
    img = np.array(img)
    out_awb = deep_wb(img, net_awb=net_awb, device=device)
    out_awb = (out_awb * 255).astype('uint8')
    plt.imshow(out_awb)
    plt.show()

    "to save"
    if tosave:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        mpimg.imsave(os.path.join(out_dir, 'cor_wb.png'),out_awb)




