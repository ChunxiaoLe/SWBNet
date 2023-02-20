"""
 The main blocks of SWBNet: Common loss function
"""

import torch


class mae_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss


class mse_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.pow((output[:, 0, :, :] - target[:, 0, :, :]), 2) + torch.pow(
            (output[:, 1, :, :] - target[:, 1, :, :]), 2)
                         + torch.pow((output[:, 2, :, :] - target[:, 2, :, :]), 2))
        # kk = output.size(0)
        return loss