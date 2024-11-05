from torch import nn
import torch
import math

#the more crazy you are, the more scale it use
# highly recommend: 1.4/0.15 & -3/0.3
# recommend:        1.2/0.12 & 1/0 & 0.8/0
# not recommend:    even (or near) shape
class SSSLU(nn.Module):
    def __init__(self,scale:float=1,bias:float=0):
        super(SSSLU, self).__init__()
        self.act1=torch.nn.Softsign()
        self.scale=scale*torch.pi/2
        self.bias=bias+1
    def forward(self, x):
        return x * (torch.sin(self.act1(x)*self.scale)+self.bias)

class SSSLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1,1], kernel_size=[3,3], padding=[1,1], sssluShape:float =1.0 ,sssluBias:float=0,bias=False,groups=1):
        super(SSSLUConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = SSSLU(sssluShape,sssluBias)

    def forward(self, x):
        x=self.conv(x)
        x = self.bn(x)
        x=self.act(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1,1], kernel_size=[3,3], padding=[1,1], bias=False,groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        #self.act = nn.SiLU(inplace=False)
        self.act = SSSLU(-3,0.3)
        #self.act2= SSSLUp(11,3.5)
        # act=silu:             times: 30   accuracy: 0.67978   loss: 1.269569  lr: 0.001028945 delta=0.0784
        # act=SSSLU(1.2/0.12):  times: 30   accuracy: 0.68615   loss: 1.243340  lr: 0.001028945 delta=0.0730
        # act=SSSLU(1.4/0.2):   times: 30   accuracy: 0.70687   loss: 1.152956  lr: 0.001028945 delta=0.0754
        # act=SSSLU(0.8):       times: 30   accuracy: 0.64539   loss: 1.396503  lr: 0.001028945 delta=0.0618
        # act=SSSLU():          times: 30   accuracy: 0.69853   loss: 1.189982  lr: 0.001028945 delta=0.7220
        # act=Mish:             times: 30   accuracy: 0.66875   loss: 1.304177  lr: 0.001028945 delta=0.0660
    def forward(self, x):
        x=self.conv(x)
        x=self.act(x)
        x = self.bn(x)
        return x

# middle rank conv 3*width (not optimized)
class MRConv3(nn.Module):
    # c1,c2,pooling
    def __init__(self, in_channels, out_channels,stride, bias=False):
        super(MRConv3, self).__init__()
        groups=math.gcd(in_channels,out_channels)
        self.l1wc=Conv(in_channels,in_channels,[1,stride],[1,3],[0,1])
        self.l1hc=Conv(in_channels,in_channels,[stride,1],[3,1],[1,0])
        self.l1wh=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            #SSSLU(1.4,0.2)
            nn.SiLU()
        )
        self.l2wc=Conv(out_channels,out_channels,[1,stride],[1,3],[0,1])
        self.l2hc=Conv(in_channels,in_channels,[stride,1],[3,1],[1,0])
        self.l2wh=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            #SSSLU(1.4,0.2)
            nn.SiLU()
        )
        self.l3wc=Conv(out_channels,out_channels,[1,stride],[1,3],[0,1])
        self.l3hc=Conv(out_channels,out_channels,[stride,1],[3,1],[1,0])
        self.l3wh=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            #SSSLU(1.4,0.2)
            nn.SiLU()
        )
        self.ocv=Conv(out_channels*3,out_channels,1,1,0)

    def forward(self, x):
        x1=self.l1wc(x)
        x1=self.l2hc(x1)
        x1=self.l3wh(x1)
        x2=self.l1hc(x)
        x2=self.l2wh(x2)
        x2=self.l3wc(x2)
        x3=self.l1wh(x)
        x3=self.l2wc(x3)
        x3=self.l3hc(x3)
        return self.ocv(torch.cat([x1,x2,x3],1))

# middle rank conv (not optimized)
# in replace of 2x conv (sequencial) in size
class MRConv(nn.Module):
    # c1,c2,pooling
    def __init__(self, in_channels, out_channels,stride,kernel_size:int=3, bias=False):
        super(MRConv, self).__init__()
        groups=math.gcd(in_channels,out_channels)
        self.wh = Conv(in_channels, in_channels, 1, kernel_size, kernel_size // 2, bias, groups)
        self.wc = Conv(in_channels,(in_channels+out_channels)//2,(1,stride),(1,kernel_size),(0,kernel_size//2),bias=bias)
        self.hc = Conv((in_channels+out_channels)//2, out_channels, (stride, 1), (kernel_size, 1), (kernel_size // 2, 0), bias=bias)

    def forward(self,x):
        x=self.wh(x)
        x=self.wc(x)
        x=self.hc(x)
        return x

class MRCn(nn.Module):
    def __init__(self, in_channels, out_channels,depth:int=2,e:float=0.5, bias=False):
        super(MRCn, self).__init__()
        self.hid=int(in_channels*e)
        self.resh=Conv(in_channels,self.hid*2,1,bias=bias)
        self.cv1=nn.ModuleList([MRConv(self.hid,self.hid,1,bias=bias) for _ in range(depth)])
        self.createdByYrx=Conv((depth+1)*self.hid,out_channels,stride=1,bias=bias)

    def forward(self,x:torch.Tensor):
        y=list(torch.chunk(self.resh(x),2,1))
        for cv in self.cv1:
            y.append(cv(y[-1]))
        return self.createdByYrx(torch.cat(y,1))

class MRCnf1(nn.Module):
    def __init__(self, in_channels, out_channels,depth:int=2,e:float=0.5, bias=True):
        super(MRCnf1, self).__init__()
        self.hid=int(in_channels*e)
        self.resh=Conv(in_channels,self.hid*2,1,bias=bias)
        self.down=Conv(self.hid,self.hid,2,bias=bias)
        self.cv1=nn.ModuleList([MRConv(self.hid,self.hid,1,bias=bias) for _ in range(depth)])
        self.createdByYrx=Conv((depth+2)*self.hid,out_channels,stride=1,bias=bias)

    def forward(self,x:torch.Tensor):
        x, x1=torch.chunk(self.resh(x),2,1)
        y=[self.down(x1)]
        for cv in self.cv1:
            y.append(cv(y[-1]))
        return self.createdByYrx(torch.cat([x,nn.functional.interpolate((torch.cat(y,1)),(x.size(2),x.size(3)),mode='nearest')],1))


# low rank conv (not optimized)
class LRConv(nn.Module):
    # c1,c2,pooling
    def __init__(self, in_channels, out_channels, stride,kernel_size:int=3, bias=False):
        super(LRConv, self).__init__()
        groups = math.gcd(in_channels, out_channels)
        self.w = Conv(in_channels, in_channels, (1, stride), (1, kernel_size), (0, kernel_size//2), groups=groups,bias=bias)
        self.h = Conv(in_channels, in_channels, (stride, 1), (kernel_size, 1), (kernel_size//2, 0), groups=groups,bias=bias)
        self.c = Conv(in_channels,out_channels,1,1,0,bias=bias)

    def forward(self, x):
        x = self.w(x)
        x = self.h(x)
        x = self.c(x)
        return x

class LRCnf1(nn.Module):
    def __init__(self, in_channels, out_channels,depth:int=1,e:float=0.25, bias=True):
        super(MRCnf1, self).__init__()
        self.hid=int(in_channels*e)
        self.resh=LRConv(in_channels,self.hid*2,1,bias=bias)
        self.down=LRConv(self.hid,self.hid,2,bias=bias)
        self.cv1=nn.ModuleList([LRConv(self.hid,self.hid,1,bias=bias) for _ in range(depth)])
        self.createdByYrx=Conv((depth+2)*self.hid,out_channels,stride=1,bias=bias)

    def forward(self,x:torch.Tensor):
        x, x1=torch.chunk(self.resh(x),2,1)
        y=[self.down(x1)]
        for cv in self.cv1:
            y.append(cv(y[-1]))
        return self.createdByYrx(torch.cat([x,nn.functional.interpolate((torch.cat(y,1)),(x.size(2),x.size(3)),mode='nearest')],1))


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*2, channel*2 // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel*2 // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y2=self.max_pool(x).view(b,c)
        y = self.fc(torch.cat([y,y2],1)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size=(k, k), stride=(1, 1), padding=((k - 1) // 2, (k - 1) // 2),
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v

class CBAMBlock(nn.Module):
    def __init__(self, channels: int, spatial_attention_kernel_size: int = 3):
        super(CBAMBlock, self).__init__()
        self.channel_attention_block = SEBlock(channel=channels)
        self.spatial_attention_block = Spatial_Attention_Module(k=spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x

# return a deeped tensor doubled shape, depend on how you use it
# real depth is 2 X depth, means 4^n width or height
# this is faster than QuickDeep, using lower rank conv
# and there is an optional SEBlock-CBAMBlock to use.
class QuickDeepf1(nn.Module):
    def __init__(self, in_channels,depth=2,use_SEBlock=False):
        super(QuickDeepf1, self).__init__()
        self.deepen0=LRConv(in_channels,in_channels*2,2,bias=True)
        self.deepen=nn.ModuleList([LRConv(in_channels*2,in_channels*2,2,bias=False) for _ in range(depth*2-1)])
        self.wake=nn.Sequential(
            Conv(in_channels * 2 * depth, in_channels, 1, 1, 0, True),
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        if(use_SEBlock):
            self.cbam=CBAMBlock(in_channels*2,3)
        self.depth=depth
        self.simple_attn=use_SEBlock
    def forward(self, x):
        #nn.functional.interpolate(mask,[x[1].size(2),x[1].size(3)],mode='nearest')
        w = x.size(2)
        h = x.size(3)
        y = self.deepen0(x)
        startGroup=[y]
        for cv in self.deepen:
            startGroup.append(cv(startGroup[-1]))
        endGroup=list()
        for a in range(0,self.depth):
            t = nn.functional.interpolate(startGroup[a+1],[w//2,h//2],mode='nearest')
            t = t + nn.functional.interpolate(startGroup[a], [w // 2, h // 2], mode='nearest')
            endGroup.append(t)
        y = torch.cat(endGroup,1)
        y = self.wake(y)
        if(self.simple_attn):
            x=torch.cat([x,y],1)
            return self.cbam(x)
        else:
            return torch.cat([x,y],1)


# return the same size
# question size==0 --> the same channel size while quiring
class MaxPoolingAttn(nn.Module):
    def __init__(self, channels: int, question_size: int=0,use_hard:bool =False):
        if(question_size==0):
            question_size=channels
        super(MaxPoolingAttn, self).__init__()
        self.qkv=Conv(channels,question_size*2+channels,1,1,0,False)
        self.co=SSSLUConv(question_size+channels,question_size,1,3,1)#correct
        self.back=nn.Sequential(
            Conv(question_size,channels,1,3,1,True),
            SSSLUConv(channels,channels,1,1,0,0.8,0,bias=True),
            nn.Sigmoid()
            #maybe an optional SEBlock?
        )
        if(use_hard):
            self.back=nn.Sequential(
                SSSLUConv(question_size+channels,channels,1,3,1,1.2,0.12,bias=True),
                SSSLUConv(channels,channels,1,1,0,0.8,0)
            )
        self.ch=channels
        self.useHard=use_hard
        self.question_size=question_size

    def forward(self, x):
        height=x.size(2)
        width=x.size(3)
        batch=x.size(0)

        xq,xk,xv=self.qkv(x).split([self.question_size,self.question_size,self.ch],1)

        wp=nn.AdaptiveMaxPool2d((1,width))
        #wp=nn.MaxPool2d([height,1],1)
        #hp=nn.AdaptiveMaxPool2d((1,x.size(2)))#注意转置
        #hp=nn.MaxPool2d((width,1),1)

        xqw=wp(xq).view(self.question_size*batch,width,1)
        xkw=wp(xk).view(self.question_size*batch,1,width)
        #xqh=hp(xq.permute(0,1,3,2)).view(self.ch*batch,height,1).permute(0,2,1)
        #xkh=hp(xk.permute(0,1,3,2)).view(self.ch*batch,1,height).permute(0,2,1)
        #xq=xq.permute(0,1,3,2)
        #xk=xk.permute(0,1,3,2)
        #xqh = hp(xq).view(self.ch * batch, height, 1).permute(0,2,1)
        #xkh = hp(xk).view(self.ch * batch, 1, height).permute(0,2,1)
        xqh=torch.max(xq,3,keepdim=True)[0].view(self.question_size*batch,height,1)
        xkh=torch.max(xk,3,keepdim=True)[0].view(self.question_size*batch,1,height)


        xaw=torch.bmm(xqw,xkw)
        xah=torch.bmm(xqh,xkh)

        awp=nn.AdaptiveMaxPool2d((1,width))
        ahp=nn.AdaptiveMaxPool2d((1,height))
        xaw=awp(xaw)
        xah=ahp(xah.permute(0,2,1)).permute(0,2,1)

        layer_normW = nn.Sequential(
            nn.LayerNorm(normalized_shape=(1, xaw.size(2)), device=xaw.device,dtype=xaw.dtype),
            nn.SiLU()
        )
        layer_normH = nn.Sequential(
            nn.LayerNorm(normalized_shape=(xah.size(1), 1), device=xah.device,dtype=xah.dtype),
            nn.SiLU()
        )
        #xaw=layer_normW(xaw).expand(xaw.size(0),height,width)
        #xah=layer_normH(xah).expand(xah.size(0),height,width)
        xaw=layer_normW(xaw)#.repeat(1,height,1)
        xah=layer_normH(xah)#.repeat(1,1,width)

        xa=xah * xaw
        xa=xa.view([x.size(0),self.question_size,x.size(2),x.size(3)])
        xa=self.co(torch.cat([xa,xv],1))
        if(self.useHard):
            xa=self.back(torch.cat([xa,x],1))
        else:
            xa=self.back(xa)
            x=x*xa
        return xa

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))



