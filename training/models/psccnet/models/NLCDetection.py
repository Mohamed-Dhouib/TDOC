import torch
import torch.nn as nn
import torch.nn.functional as F
from .seg_hrnet_config import get_hrnet_cfg


class NonLocalMask(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMask, self).__init__()

        self.r = reduce_scale

        # input channel number
        self.ic = in_channels * self.r * self.r

        # middle channel number
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)

        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)
        self.W_c = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))
        self.gamma_c = nn.Parameter(torch.ones(1))

        # -------- CHANGE 1: always output 2-channel logits (no sigmoid) --------
        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        )
        # ----------------------------------------------------------------------

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

        # g x
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta
        theta_x = self.theta(x1).view(b, self.mc, -1)
        theta_x_s = theta_x.permute(0, 2, 1)
        theta_x_c = theta_x

        # phi x
        phi_x = self.phi(x1).view(b, self.mc, -1)
        phi_x_s = phi_x
        phi_x_c = phi_x.permute(0, 2, 1)

        # non-local attention
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)
        f_c = torch.matmul(theta_x_c, phi_x_c)
        f_c_div = F.softmax(f_c, dim=-1)

        # get y_s
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous().view(b, c, h, w)

        # get y_c
        y_c = torch.matmul(g_x, f_c_div).view(b, c, h, w)

        # get z
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)

        # get mask (no sigmoid, returns logits)
        mask = self.getmask(z.clone())
        return mask, z


class NLCDetection(nn.Module):
    def __init__(self, args):
        super(NLCDetection, self).__init__()

        self.crop_size = args['crop_size']

        FENet_cfg = get_hrnet_cfg()
        num_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']
        feat1_num, feat2_num, feat3_num, feat4_num = num_channels

        self.getmask4 = NonLocalMask(feat4_num, 1)
        self.getmask3 = NonLocalMask(feat3_num, 2)
        self.getmask2 = NonLocalMask(feat2_num, 2)
        self.getmask1 = NonLocalMask(feat1_num, 4)

    def forward(self, feat):
        s1, s2, s3, s4 = feat
        input_size = s1.shape[2:]  # original size

        # if s1.shape[2:] == self.crop_size:
        #     pass
        # else:
        #     s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
        #     s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
        #     s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
        #     s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)

        mask4, z4 = self.getmask4(s4)
        mask4U = F.interpolate(torch.softmax(mask4, dim=1)[:, 1:2], size=s3.size()[2:], mode='bilinear', align_corners=True)

        s3 = s3 * mask4U
        mask3, z3 = self.getmask3(s3)
        mask3U = F.interpolate(torch.softmax(mask3, dim=1)[:, 1:2], size=s2.size()[2:], mode='bilinear', align_corners=True)

        s2 = s2 * mask3U
        mask2, z2 = self.getmask2(s2)
        mask2U = F.interpolate(torch.softmax(mask2, dim=1)[:, 1:2], size=s1.size()[2:], mode='bilinear', align_corners=True)

        s1 = s1 * mask2U
        mask1, z1 = self.getmask1(s1)

        # -------- CHANGE 2: upsample all masks to input size --------
        mask1 = F.interpolate(mask1, size=input_size, mode='bilinear', align_corners=True)
        mask2 = F.interpolate(mask2, size=input_size, mode='bilinear', align_corners=True)
        mask3 = F.interpolate(mask3, size=input_size, mode='bilinear', align_corners=True)
        mask4 = F.interpolate(mask4, size=input_size, mode='bilinear', align_corners=True)
        # ------------------------------------------------------------

        return mask1, mask2, mask3, mask4
