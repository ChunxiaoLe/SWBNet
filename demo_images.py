"""
 Reference:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""


import argparse
import logging
import os
import torch
from PIL import Image
from arch import deep_wb_model_msa,deep_wb_single_task
from utilities.deepWB import deep_wb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def to_image(image):
    """ converts to PIL image """
    return Image.fromarray(image)



def get_args():
    parser = argparse.ArgumentParser(description='Changing WB of an input image.')
    parser.add_argument('--model_dir', '-m', default='/home/chunxiao/deep_final/PyTorch/models/',
                        help="Specify the directory of the trained model.", dest='model_dir')
    parser.add_argument('--input_dir', '-i', help='Input image directory', dest='input_dir',
                        default='/dataset/lcx/set1_all1/')
    parser.add_argument('--gt_dir', '-g', help='GT image directory', dest='gt_dir',
                        default='/dataset/lcx/set1_all1/')
    parser.add_argument('--output_dir', '-o', default='/dataset/lcx/cube/Cube_val/',
                        help='Directory to save the output images', dest='out_dir')
    parser.add_argument('--mxsize', '-S', default=656, type=int,
                        help="Max dim of input image to the network, the output will be saved in its original res.",
                        dest='S')
    parser.add_argument('--task', '-t', default='AWB',
                        help="Specify the required task: 'AWB', 'editing', or 'all'.", dest='task')
    parser.add_argument('--save', '-s', action='store_false',
                        help="Save the output images",
                        default=True, dest='save')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')

    ### dct123_att1_nofroze_pre_b5_5000new_all_5000_multinet loss_nofroze_pre_net_awb_5000_23_all_5000_23_multinet
    parser.add_argument('-model_name', '--model_name', dest='model_name',
                        default='net_ctif')

    parser.add_argument('-type', '--type', dest='type', default='test')  ## dc: 0
    ### Can6 Can1D Fuj IMG Nik40 Nik52 cube 8D5U ALL
    parser.add_argument('-cam', '--camera', dest='camera', default='mul_all',
                        help='Testing camera')


    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    "整理数据"
    # image_dir = []
    # data_dir = '/home/dataset/lcx/selected images/'
    # pos = '.jpg'
    # for fn in os.listdir(data_dir):
    #     if fn.lower().endswith(pos):
    #         image_dir.append(fn)
    #
    # np.save('/home/chunxiao/deep_final/folds/LSIM.npy', np.array(image_dir))

    save_dir = '/home/dataset/lcx/swb_selected_images/'
    if args.save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


    "输入模型"
    net_awb = deep_wb_model_msa.deepWBNet()
   # net_awb = deep_wb_single_task.deepWBnet()
    logging.info("Loading model {}".format(
        os.path.join(args.model_dir,
                     args.model_name + '.pth')))

    net_awb.load_state_dict(
        torch.load(os.path.join(args.model_dir,
                                args.model_name + '.pth')
                   , map_location='cuda:0'))
    net_awb.to(device=device)
    net_awb.eval()

    "输入数据"
    in_list = np.load('/home/chunxiao/deep_final/folds/LSIM.npy')
    data_dir = '/home/dataset/lcx/selected images/'


    for i in range(len(in_list)):
        print(in_list[i])
        in_name = data_dir + in_list[i]
        # gt_name = gt_list[i]

        in_img = Image.open(in_name)
        # gt_img = np.array(Image.open(gt_name)).astype(np.float)

        in_img = np.array(in_img)
        # plt.imshow(in_img)
        # plt.axis('off')
        # plt.show()
        out_awb = deep_wb(in_img, net_awb=net_awb, device=device)
        out_awb = (out_awb * 255).astype('uint8')
        plt.imshow(out_awb)
        plt.title(in_list[i])
        plt.axis('off')
        plt.show()

        out_awb1 = to_image(out_awb)
        out_awb1.save(save_dir + in_list[i])







