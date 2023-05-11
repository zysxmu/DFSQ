import torch
import torch.nn as nn
from model.setq_conv_quant_ops import quant_conv3x3_setq,SetQConv2d

class SetQ_RDB_Conv_in(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3, w_bits=32,a_bits=32, name=None,postReLU=False):
        super(SetQ_RDB_Conv_in, self).__init__()
        Cin = inChannels
        G = growRate


        self.conv = nn.Sequential(*[
            quant_conv3x3_setq(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1, bias=True,w_bits=w_bits,a_bits=a_bits,postReLU=postReLU),
            nn.ReLU()
        ])

    def forward(self, x, i):
        out = self.conv(x)
        return torch.cat((x, out), 1)

# 名字改为PTQ_RDB
class SetQ_RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3, w_bits=32,a_bits=32):
        super(SetQ_RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            if c==0:
                convs.append(SetQ_RDB_Conv_in(G0 + c * G, G, kSize, w_bits=w_bits,a_bits=a_bits,postReLU=False))
            else:
                convs.append(SetQ_RDB_Conv_in(G0 + c * G, G, kSize, w_bits=w_bits,a_bits=a_bits,postReLU=False))
        self.convs = nn.Sequential(*convs)

        
        self.LFF = SetQConv2d(in_channels=G0 + C * G, out_channels=G0, kernel_size=1, padding=0,
                                    stride=1, bias=True,w_bits=w_bits,a_bits=a_bits,postReLU=False)

    def forward(self, x):
        out = x
        for i, c in enumerate(self.convs):
            out = c(out, i)
        return self.LFF(out) + x

    @property
    def name(self):
        return 'rdb'

class SetQ_RDN(nn.Module):
    def __init__(self, args):
        super(SetQ_RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                SetQ_RDB(growRate0=G0, growRate=G, nConvLayers=C, w_bits=args.w_bits,a_bits=args.a_bits)
            )
        

        self.GFF = nn.Sequential(*[
            SetQConv2d(self.D * G0, G0, 1, padding=0, stride=1, bias=True,w_bits=args.w_bits,a_bits=args.a_bits,postReLU=False),
            SetQConv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1, bias=True,w_bits=args.w_bits,a_bits=args.a_bits,postReLU=False)
        ])

        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        out = x
        return self.UPNet(x), out

    @property
    def name(self):
        return 'rdn'
    
