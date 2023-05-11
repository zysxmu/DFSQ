import torch
import torch.nn as nn
import math
from model.setq_conv_quant_ops import quant_conv3x3_setq
import pdb

class _Residual_Block(nn.Module):
    def __init__(self,args):

        super(_Residual_Block, self).__init__()
        self.conv1 = quant_conv3x3_setq(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False,w_bits=args.w_bits,a_bits=args.a_bits,postReLU=False)
        self.in1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.PReLU()
        self.conv2 = quant_conv3x3_setq(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False,w_bits=args.w_bits,a_bits=args.a_bits,postReLU=False)
        self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output

class SetQ_SRResNet(nn.Module):
    def __init__(self,args):
        super(SetQ_SRResNet, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.PReLU()
        
        self.residual = self.make_layer(_Residual_Block, 16,args)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(64, affine=True)
        
        if args.scale[0] == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.PReLU(),
            )

        elif args.scale[0] == 2: 
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.PReLU(),
            )   
    
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer,args):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(args))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out1 = self.upscale(out)
        out1 = self.conv_output(out1)
        return out1,out
