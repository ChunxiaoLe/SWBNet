"""
 The main blocks of SWBNet

 Reference：
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""

from .deep_wb_blocks import *
import cv2

"CTSF extractor + CT-contrastive loss"
class deepWBNet(nn.Module):
    def __init__(self):
        super(deepWBNet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)

        self.encoder_down1 = DownBlock_res_dct1(24, 48, True)
        self.encoder_down2 = DownBlock_res_dct1(48, 96, True)
        self.encoder_down3 = DownBlock_res_dct1(96, 192, True)
        self.encoder_bridge_down = BridgeDown(192, 384, False)

        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = UpBlock(192, 96)
        self.decoder_up2 = UpBlock(96, 48)
        self.decoder_up3 = UpBlock(48, 24)
        self.decoder_out = OutputBlock(24, self.n_channels)




    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x_out = self.decoder_up3(x, x2)
        out = self.decoder_out(x_out, x1)
        return out




