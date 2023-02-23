"""
 Reference:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""


import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from arch import deep_wb_model_msa,MultiTaskLoss,deep_wb_model_mvit

import arch.splitNetworks as splitter

try:
    from torch.utils.tensorboard import SummaryWriter

    use_tb = False
except ImportError:
    use_tb = False

from utilities.dataset_msa import BasicDataset
from utilities.loss_func import mae_loss,CCLoss_inter

from torch.utils.data import DataLoader, random_split


def train_net(net,
              device,
              epochs=110,
              batch_size=32,
              lr=0.0001,
              val_percent=0.1,
              lrdf=0.5,
              lrdp=25,
              fold=0,
              chkpointperiod=1,
              trimages=12000,
              patchsz=128,
              patchnum=4,
              validationFrequency=4,
              dir_img='../dataset',
              cam = 'camera',
              save_cp=True):
    dir_checkpoint = f'checkpoints_{args.model_name}_{args.data_name}_{args.test_name}/'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
    train = BasicDataset(dir_img, name=args.data_name, patch_size=patchsz, patch_num_per_image=patchnum, type='train',cam=cam)
    val = BasicDataset(dir_img, name=args.test_name, patch_size=patchsz, patch_num_per_image=patchnum,type='test',cam=args.camera_test)
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    if use_tb:
        writer = SummaryWriter(comment=f'LR_{args.model_name}_{cam}_{args.data_name}_{args.camera_test}_{lr}_WB_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Patches/image:   {patchnum}
        Learning rate:   {lr}
        Training size:   {12000}
        Validation size: {21046}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')

    # Mloss = MultiTaskLoss.MultiTaskLossWrapper(1, net,device)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)
    # for k, v in Mloss.model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    # froze_ = [Mloss.log_vars[0],Mloss.log_vars[1]]
    # froze_net = [Mloss.model.decoder_out.down, Mloss.model.decoder_out.ford, Mloss.model.decoder_out.linear]
    # froze_net = [Mloss.model.decoder_out.mvit.regressor]

    best_val_loss = 100000

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=5188, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs_ = batch['image']
                awb_gt_ = batch['gt-AWB']
                label_ = batch['label']

                assert imgs_.shape[1] == net.n_channels * patchnum, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert awb_gt_.shape[1] == net.n_channels * patchnum, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded AWB GT images have {awb_gt_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'



                for j in range(patchnum):
                    imgs = imgs_[:, (j * 3): 3 + (j * 3), :, :]
                    awb_gt = awb_gt_[:, (j * 3): 3 + (j * 3), :, :]
                    label = label_[:, :]

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    awb_gt = awb_gt.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)

                    # imgs_pred = Mloss(imgs,awb_gt,label)
                    imgs_pred = net(imgs)
                    loss = mae_loss.compute(imgs_pred,awb_gt)

                    loss_save = open(loss_t, mode='a')
                    loss_save.write(
                        '\n' + 'epoch:' + str(epoch)  + '  loss:' + str(loss) )
                    # 关闭文件
                    loss_save.close()
                    epoch_loss += loss.item()

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # if epoch < args.ep_s:
                    #     loss1.backward(retain_graph=True)
                    # else:
                    #     loss.backward(retain_graph=True)
                    # loss.backward(retain_graph=True)
                    optimizer.step()
                    pbar.update(np.ceil(imgs.shape[0] / patchnum))
                    global_step += 1

        if (epoch + 1) % validationFrequency == 0:
            if use_tb:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score = vald_net(net, val_loader, device)
            loss_save = open(val_t, mode='a')
            loss_save.write(
                '\n' + 'epoch:' + str(epoch) + '  val:' + str(val_score))
            # 关闭文件
            loss_save.close()
            logging.info('Validation MAE: {}'.format(val_score))

            if 0 < val_score < best_val_loss:
                best_val_loss = val_score
                print("Saving new best model... \n")
                torch.save(net.state_dict(),
                           'models/' + args.model_name + '_' + args.data_name + '_' + args.camera + 'net.pth')
                logging.info('Saved trained model!')

            if use_tb:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Loss/test', val_score, global_step)
                writer.add_images('images', imgs, global_step)
                writer.add_images('result-awb', imgs_pred[:, :3, :, :], global_step)
                # writer.add_images('result-t', imgs_pred[:, 3:6, :, :], global_step)
                # writer.add_images('result-s', imgs_pred[:, 6:, :, :], global_step)
                writer.add_images('GT_awb', awb_gt, global_step)
                # writer.add_images('GT-t', t_gt, global_step)
                # writer.add_images('GT-s', s_gt, global_step)

        scheduler.step()

    logging.info('Saved trained models!')
    if use_tb:
        writer.close()
    logging.info('End of training')


def vald_net(net, loader, device):
    """Evaluation using MAE"""
    net.eval()
    n_val = len(loader) + 1
    mae = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs_ = batch['image']
            awb_gt_ = batch['gt-AWB']
            label_ = batch['label']

            patchnum = imgs_.shape[1] / 3
            assert imgs_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.model.n_channels} input channels, ' \
                f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            assert awb_gt_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.model.n_channels} input channels, ' \
                f'but loaded AWB GT images have {awb_gt_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'


            imgs = imgs_[:, 0:3, :, :]
            awb_gt = awb_gt_[:, 0:3, :, :]
            label = label_[:, :]

            imgs = imgs.to(device=device, dtype=torch.float32)
            awb_gt = awb_gt.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                imgs_pred = net(imgs)
                # _, _, _, loss1, _, _ = net(imgs, awb_gt, label)
                # imgs_pred,_ = net(imgs)
                loss1 = mae_loss.compute(imgs_pred, awb_gt)
                mae = mae + loss1

            pbar.update(np.ceil(imgs.shape[0] / patchnum))

    net.train()
    return mae / n_val


def get_args():
    parser = argparse.ArgumentParser(description='Train deep WB editing network.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=180,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq', type=int, default=5,
                        help='Validation frequency.')
    parser.add_argument('-d', '--fold', dest='fold', type=int, default=1,
                        help='Testing fold to be excluded. Use --fold 0 to use all Set1 training data')
    parser.add_argument('-p', '--patches-per-image', dest='patchnum', type=int, default=4,
                        help='Number of training patches per image')
    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int, default=128,
                        help='Size of training patch')
    parser.add_argument('-t', '--num_training_images', dest='trimages', type=int, default=9000,
                        help='Number of training images. Use --num_training_images 0 to use all training images')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=10,
                        help='Number of epochs to save a checkpoint')
    parser.add_argument('-ldf', '--learning-rate-drop-factor', dest='lrdf', type=float, default=0.5,
                        help='Learning rate drop factor')
    parser.add_argument('-ldp', '--learning-rate-drop-period', dest='lrdp', type=int, default=25,
                        help='Learning rate drop period')
    parser.add_argument('-trd', '--training_dir', dest='trdir', default='',
                        help='Training image directory')
    ### Can6 Can1D Fuj IMG Nik40 Nik52 cube 8D5U ALL
    parser.add_argument('-cam', '--camera', dest='camera', default='multi',
                        help='Training camera')
    parser.add_argument('-cat', '--camera-t', dest='camera_test', default='multi',
                        help='Training camera')
    #### multi-camera 直接给名字
    parser.add_argument('-dan', '--data-name', dest='data_name', default='all_12000_12',
                        help='Training camera')
    parser.add_argument('-tsn', '--test-name', dest='test_name', default='cc_sm',
                       help='Training camera')
    ### res-1_with_se
    parser.add_argument('-mon', '--model-name', dest='model_name', default='test_dct123_ctm_net_awb',
                        help='Training camera')
    parser.add_argument('-ep_s', '--ep_s', dest='ep_s', type=int, default=0,
                        help='Training camera')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    logging.info(f'Training of _{args.model_name}_{args.data_name}_{args.test_name}')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = deep_wb_model_msa.deepWBNet(args.patchsz,args.patchsz,'dct')

    if args.load:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(args.load, map_location=device)
        # net_cdct_19.pth
        logging.info(f'loading Stage1')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

        net.to(device=device)

    net.to(device=device)



    for k, v in net.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))


    loss_t = './loss/camera_'+ args.model_name +'_' + args.data_name +'_'+ args.camera + '_train.txt'
    val_t = './loss/camera_' + args.model_name +'_' + args.data_name +'_'+ args.camera + '_test.txt'

    try:
        f = open(loss_t, 'r')
        f.close()
    except IOError:
        f = open(loss_t, 'w')

    try:
        f = open(val_t, 'r')
        f.close()
    except IOError:
        f = open(val_t, 'w')

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  lrdf=args.lrdf,
                  lrdp=args.lrdp,
                  device=device,
                  fold=args.fold,
                  chkpointperiod=args.chkpointperiod,
                  trimages=args.trimages,
                  val_percent=args.val / 100,
                  validationFrequency=args.val_frq,
                  patchsz=args.patchsz,
                  patchnum=args.patchnum,
                  dir_img=args.trdir,
                  cam = args.camera
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
