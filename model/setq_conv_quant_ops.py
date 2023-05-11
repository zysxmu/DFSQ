import torch
import torch.nn.functional as F
import torch.nn as nn
from model.quant_ops import TorchRound




class SetQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', w_bits=32, a_bits=32, channel_wise=True, postReLU=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode)

        self.w_bits = w_bits
        self.w_qmax = 2**w_bits-1
        self.w_interval = None
        self.w_zpoint = None
        self.channel_wise = channel_wise
        self.postReLU = postReLU

        if self.channel_wise:
            self.w_clip_upper = nn.Parameter(torch.zeros((out_channels, 1, 1, 1)))
            self.w_clip_lower = nn.Parameter(torch.zeros((out_channels, 1, 1, 1)))
        else:
            self.w_clip_upper = nn.Parameter(torch.zeros((1,1,1,1)))
            self.w_clip_lower = nn.Parameter(torch.zeros((1, 1, 1, 1)))

        self.a_bits = a_bits
        self.a_qmax = 2**(a_bits-1)
        self.qps = None
        
        self.round = TorchRound()
        self.mode = "raw"
        self.raw_input = None
        self.raw_output = None
        self.quant_output = None

    def subset_quant(self, x, qps):  
        x = x.cuda()
        d = x.shape
        x_ = x.transpose(0, 1).reshape(d[1], -1)
       
        x_q = []
        for channel in range(self.in_channels):
            qps_ = qps[channel]
            inp = x_[channel]
            x_q_ = qps_[torch.min( torch.abs( (inp[:, None]).expand(len(inp), len(qps_)) - (qps_[None, :]).expand(len(inp), len(qps_)) ), 1)[1]]
            x_q.append(x_q_)
        x_q = torch.stack(x_q)
        return (torch.reshape(x_q, (d[1],d[0],d[2],d[3])).transpose(0, 1) - x).detach() + x


    
    def quantize(self,x):
        return self.subset_quant(x, self.qps)

    def quant_input(self, x):
        d = x.size() 

        mu_gt = torch.mean(x, (2, 3), True)
        mu = mu_gt.detach()
        maxv_gt = torch.max(torch.abs(x).view(
            d[0], d[1], -1), dim = -1, keepdim=True)[0].view(d[0], d[1], 1, 1)
        maxv = maxv_gt.detach()

        if self.postReLU == True:
            x_ = x / (maxv + 1e-8)
        else:
            x = (x - mu) 
            maxv_gt = torch.max(torch.abs(x).view(
                d[0], d[1], -1), dim = -1, keepdim=True)[0].view(d[0], d[1], 1, 1)
            maxv = maxv_gt.detach()
            x_ = x / (maxv + 1e-8)

        x_q = self.quantize(x_)
        x_q = x_q 

        if self.postReLU == True:
            x_q = x_q * (maxv + 1e-8)
        else:
            x_q = x_q * (maxv + 1e-8) + mu            

        return x_q
    
    def quant_weight(self):
        self.w_interval = ((self.w_clip_upper-self.w_clip_lower)/self.w_qmax).view(-1, 1, 1, 1)
        self.w_zpoint = self.round(self.w_clip_lower/self.w_interval) + 2**(self.w_bits-1)
        q_weight = (self.round(self.weight/self.w_interval) - self.w_zpoint).clamp_(-(self.w_qmax+1)/2, (self.w_qmax+1)/2-1)
        r_weight = (q_weight+self.w_zpoint)*self.w_interval
        return r_weight

    def init_params(self):
        if self.channel_wise:
            w = self.weight.reshape(self.out_channels, -1)
        else:
            w = self.weight.reshape(1, -1)
        max_val = (torch.max(w, dim=-1, keepdim=True)[0]).view(-1, 1, 1, 1)
        min_val = (torch.min(w, dim=-1, keepdim=True)[0]).view(-1, 1, 1, 1)
        self.w_interval = ((max_val-min_val)/self.w_qmax).view(-1, 1, 1, 1)
        self.w_zpoint = (self.round(min_val/self.w_interval) + 2**(self.w_bits-1)).view(-1, 1, 1, 1)
        for i in range(self.out_channels):
            self.w_clip_upper.data[i].fill_(max_val[i].squeeze().data)
            self.w_clip_lower.data[i].fill_(min_val[i].squeeze().data)
    def calibration_step(self): 
        self.init_params()

    def forward(self, x):
        # FP
        if self.mode == "raw":
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant":  
            return self.quant_forward(x)
        elif self.mode == "step2":
            return self.calibration_step2(x)

    def quant_forward(self, x):
        if self.a_bits < 32:
            r_x = self.quant_input(x)
        else:
            r_x = x

        if self.w_bits < 32:
            r_w = self.quant_weight()
        else:
            r_w = self.weight
        r_out = F.conv2d(r_x, r_w, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        return r_out


def quant_conv3x3_setq(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, w_bits=32, a_bits=32, channel_wise=True,postReLU=False):
    return SetQConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, w_bits=w_bits, a_bits=a_bits, channel_wise=channel_wise, postReLU=postReLU)
