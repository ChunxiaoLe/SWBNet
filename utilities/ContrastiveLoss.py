"""
 The main blocks of SWBNet: CT-contrastive loss

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:1', temperature=0.7):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))# 超参数 温度

        ## negatives mask
        # self.batch_size = batch_size
        self.sigle_eye = ~torch.eye(batch_size,dtype=torch.bool)
        self.sigle_eye1 = torch.cat((self.sigle_eye,self.sigle_eye),dim=0)
        self.eye = torch.cat((self.sigle_eye1,self.sigle_eye1),dim=1)
        self.register_buffer("negatives_mask", (self.eye.to(device)).float())

        # 主对角线为0，其余位置全为1的mask矩阵
        self.all = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)
        self.register_buffer("all_mask", (self.all.to(device)).float())


    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        # z_i = emb_i
        # z_j = emb_j

        representations = torch.cat([z_i, z_j], dim=0)
        # r1 = representations.unsqueeze(1)
        # r2 = representations.unsqueeze(0)# repre: (2*bs, dim)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
        #                                         dim=2)  # simi_mat: (2*bs, 2*bs)
        #
        # s1 = similarity_matrix.cpu().detach().numpy()

        x = representations

        # xp11  = torch.pow(x, 2)
        # xp12 = torch.sum(xp11,dim=1,keepdim=True)
        # xp13 = xp12.expand([self.batch_size*2, self.batch_size*2])

        xp1 = torch.pow(x, 2).sum(dim=1, keepdim=True).expand([self.batch_size*2, self.batch_size*2])
        xp2 = xp1.transpose(0, 1)
        xg = torch.mm(x, x.transpose(0, 1))
        similarity_matrix = xp1 + xp2 - xg

        # s2 = similarity_matrix.cpu().detach().numpy()

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature) # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)   # 2*bs, 2*bs

        loss_partial1 = -torch.log(nominator / (torch.sum(denominator, dim=1))+ torch.exp(positives))  # 2*bs
        loss = torch.sum(loss_partial1) / (2 * self.batch_size)


        # all_dis = self.all_mask * torch.exp(torch.sum(similarity_matrix))
        # loss_partial2 = -torch.log(all_dis)
        return loss




