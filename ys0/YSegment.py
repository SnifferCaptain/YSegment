import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1,1], kernel_size=[3,3], padding=[1,1], bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(negative_slope=0.001,inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# return a deeped tensor same shape, depend on how you use it
# real depth is 2 X depth, means 4^n width or height
class QuickDeep(nn.Module):
    def __init__(self, in_channels,depth=2):
        super(QuickDeep, self).__init__()
        self.deepen0=Conv(in_channels,in_channels*2,2)
        self.deepen=nn.ModuleList([Conv(in_channels*2,in_channels*2,2) for _ in range(depth*2-1)])
        self.wake=Conv(in_channels*2*depth,in_channels,1,1,0,True)
        self.depth=depth
    def forward(self, x):
        #nn.functional.interpolate(mask,[x[1].size(2),x[1].size(3)],mode='nearest')
        w=x.size(2)
        h=x.size(3)
        x=self.deepen0(x)
        startGroup=list(x)
        for cv in self.deepen:
            startGroup.append(cv(startGroup[-1]))
        endGroup=list()
        for a in range(0,self.depth):
            t = nn.functional.interpolate(startGroup[a+1],[w/2,h/2],mode='nearest')
            t = t + nn.functional.interpolate(startGroup[a], [w / 2, h / 2], mode='nearest')
            endGroup.append(t)
        x=torch.cat(endGroup,1)
        x=self.wake(x)
        return x

class YSegment0(nn.Module):
    def __init__(self, in_channels, num_classes,HWC_input=True):
        super(YSegment0, self).__init__()
        self.hwc=HWC_input
        self.in0=Conv(in_channels,32,bias=True)
        self.deep0=QuickDeep(32,3)#  /64
        self.deep1=QuickDeep(64,2)#  /16
        self.out0=nn.Sequential(
            Conv(128,128,bias=True),
            Conv(128,256),
            Conv(256,num_classes,1,1,0)
        )
    def forward(self, x):
        if(self.hwc):
            x=torch.permute(x,[2,0,1])
        x=self.in0(x)
        x=torch.cat([self.deep0(x),x],1)
        x=torch.cat([self.deep1(x),x],1)
        return self.out0(x)
