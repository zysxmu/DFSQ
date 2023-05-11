

import time
import torch.nn as nn
from model import common

from model.quant_ops import conv3x3
from model.setq_conv_quant_ops import quant_conv3x3_setq
from model import common


class SetQ_ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=False, 
                bn=False, act=nn.ReLU(False), res_scale=1, w_bits = 32,a_bits=32, ema_epoch=1, name=None):
        super(SetQ_ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias,w_bits=w_bits,a_bits=a_bits,postReLU=False))
                m.append(act)
            elif i == 1:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias,w_bits=w_bits, a_bits=a_bits, postReLU=True))
                
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.shortcut = common.ShortCut()
        
    def forward(self, x):
        residual = self.shortcut(x)
        body = self.body(x).mul(self.res_scale)
        res = body
        res += residual

        return res

class SetQ_EDSR(nn.Module):
    def __init__(self, args, conv=quant_conv3x3_setq, bias=False):
        super(SetQ_EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv3x3(args.n_colors, n_feats, kernel_size,  bias=bias)]

        m_body = [
            SetQ_ResBlock(
                quant_conv3x3_setq, n_feats, kernel_size, act=act, res_scale=args.res_scale, bias=bias, w_bits=args.w_bits, a_bits=args.a_bits
            ) for i in range(n_resblock)
        ]

        m_body.append(conv3x3(n_feats, n_feats, kernel_size, bias= bias))

        m_tail = [
            common.Upsampler(conv3x3, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=kernel_size//2
            )
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        out = res
        x = self.tail(res)
        x = self.add_mean(x)
        return x, out   
    
    @property
    def name(self):
        return 'edsr'
