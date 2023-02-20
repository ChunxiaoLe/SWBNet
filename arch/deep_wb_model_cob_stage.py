"""
 The main blocks of SWBNet

 Reference：
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""

from .deep_wb_blocks import *


class deepWBNet(nn.Module):
    def __init__(self,hight,wide,type):
        super(deepWBNet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)

        if type == 'base':
            self.encoder_down1 = DownBlock(24, 48, False)
            self.encoder_down2 = DownBlock(48, 96, False)
            self.encoder_down3 = DownBlock(96, 192, False)
            self.encoder_bridge_down = BridgeDown(192, 384, False)

        elif type == 'ctif':
            self.encoder_down1 = DownBlock_res_dct3(24, 48, True)
            self.encoder_down2 = DownBlock_res_dct3(48, 96, True)
            self.encoder_down3 = DownBlock_res_dct3(96, 192, True)
            self.encoder_bridge_down = BridgeDown(192, 384, False)

        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = UpBlock(192, 96)
        self.decoder_up2 = UpBlock(96, 48)
        self.decoder_up3 = UpBlock(48, 24)

        # self.decoder_out1 = DoubleConvBlock(48, 24)
        # self.decoder_out2 = nn.Conv2d(24, 3, kernel_size=1)

        self.decoder_out = OutputBlock(24, self.n_channels)

        # self.stage2 = Stage2_mixed_new1(in_channels=self.n_channels, hight=hight, wide=wide, patch_size=8,lay=2,heads=16)
        self.stage2 = CTSTRANS(in_channels=self.n_channels, hight=hight, wide=wide, patch_size=8, lay=2,
                                        heads=16)

    def forward(self, x):
        #### stage1
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x6 = self.decoder_bridge_up(x5)
        x7 = self.decoder_up1(x6, x4)
        x8 = self.decoder_up2(x7, x3)
        x9 = self.decoder_up3(x8, x2)

        out_s1 = self.decoder_out(x9, x1)
        #### stage2

        out_s2,att = self.stage2(out_s1)
        #### DC
        return out_s2




