import torch
from torch import nn
from Yblock import *



class Ynet2B1(nn.Module):
    def __init__(self):
        super(Ynet2B1, self).__init__()
        self.l0 = Conv(3, 16, 2)  # 16-320
        self.l1 = QuickDeepf1(16, 2, True)
        self.l2 = Conv(32,32,2)  # 32-160
        self.l3 = MRCnf1(32, 48, 1, e=0.25)  # 64-320
        self.l4 = Conv(48, 48, 2, bias=True)  # 64-160
        self.l5 = MRCnf1(48, 64, 1, e=0.25)
        self.l6 = MaxPoolingAttn(64, 8, False)  # 128-160

    def forward(self, x):
        # x = x.transpose(1, 3).contiguous() #same as yolo
        return self.l6(self.l5(self.l4(self.l3(self.l2(self.l1(self.l0(x)))))))

class Ynet2B2(nn.Module):
    def __init__(self):
        super(Ynet2B2, self).__init__()
        self.l5 = Conv(64, 96, 2, bias=True)  # 256-40
        self.l6 = MRCnf1(96, 96, 1,e=0.325)
        self.l7 = MaxPoolingAttn(96, 16, True)  # 256-80

    def forward(self, x):
        return self.l7(self.l6(self.l5(x)))

class Ynet2B3(nn.Module):
    def __init__(self):
        super(Ynet2B3, self).__init__()
        self.l8 = Conv(96, 192, 2, bias=True)  # 512-40
        self.l9 = MRCnf1(192, 192, 3, e=0.75)  # 512-80
        self.l10 = SPPF(192,192)

    def forward(self, x):
        return self.l10(self.l9(self.l8(x)))

# from down to up, first deep
class Ynet2Hu1(nn.Module):
    def __init__(self):
        super(Ynet2Hu1, self).__init__()
        self.h1=nn.Upsample(scale_factor=2,mode='nearest')
        self.h2=SEBlock(288)
        self.h3=MRCnf1(288,128,2,e=0.25)

    def forward(self, x):
        return self.h3(self.h2(torch.cat([x[1],self.h1(x[0])],1)))

# from down to up, first deep
class Ynet2Hu2(nn.Module):
    def __init__(self):
        super(Ynet2Hu2, self).__init__()
        self.h1=nn.Upsample(scale_factor=2,mode='nearest')
        self.h2=SEBlock(192)
        self.h3=LRConv(192,96,1)

    def forward(self, x):
        return self.h3(self.h2(torch.cat([x[1],self.h1(x[0])],1)))

# first shallow
class Ynet2Hd1(nn.Module):
    def __init__(self):
        super(Ynet2Hd1, self).__init__()
        self.h1=Conv(96,96,2,bias=True)
        self.h2=SEBlock(224)
        self.h3=MRCnf1(224,128,1,e=0.25)

    def forward(self, x):
        return self.h3(self.h2(torch.cat([x[1],self.h1(x[0])],1)))

# first shallow
class Ynet2Hd2(nn.Module):
    def __init__(self):
        super(Ynet2Hd2, self).__init__()
        self.h1=Conv(128,128,2,bias=True)
        self.h2=LRConv(320,192,1,bias=True)

    def forward(self, x):
        return self.h2(torch.cat([x[1],self.h1(x[0])],1))
