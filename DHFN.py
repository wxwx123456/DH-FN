import torch
from torchvision import models as resnet_model
from torch import nn
# from transformer import TransformerModel
from torchvision.models import swin_b, Swin_B_Weights
import math
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PGF(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(PGF,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU(inplace=True)
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x*psi
    
class PGF1(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(PGF1,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU(inplace=True)
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return (x*0.5+g*0.5)*psi

class PGF2(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(PGF2,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU(inplace=True)
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return (x*0.8+g*0.2)*psi

class CDC_R(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CDC_R, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class HFA(nn.Module):
    def __init__(self, channel):
        super(HFA, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class CA(nn.Module):
    def __init__(self, channel, r=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DHF(nn.Module):
    def __init__(self, num_classes=9):
        super(DHF, self).__init__()
        self.num_classes = num_classes

        resnet = resnet_model.resnet34(weights=resnet_model.ResNet34_Weights.DEFAULT) # pretrained = True
        self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1, progress=True)


###############################resnet
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
##############################
        self.CA_1 = CA(256)
        self.CA_2 = CA(512)
        self.CA_3 = CA(1024)

        self.cdc_1 = CDC_R(256, 32)
        self.cdc_2 = CDC_R(512, 32)
        self.cdc_3 = CDC_R(1024, 32)

        self.hfa1 = HFA(32)
        self.down = nn.MaxPool2d(4)
        self.conv = BasicConv2d(1, 1024, 1)

        self.PG3 = PGF2(F_g=512,F_l=512,F_int=256)#大核注意力机制
        self.PG2 = PGF1(F_g=256,F_l=256,F_int=128)
        self.PG1 = PGF(F_g=1024,F_l=1024,F_int=512)


        self.decoder1 = DecoderBottleneckLayer(256)
        self.decoder2 = DecoderBottleneckLayer(512)
        self.decoder3 = DecoderBottleneckLayer(1024)


        self.up3_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4)
        self.out4 = nn.Conv2d(1, num_classes,kernel_size=1)
        self.out3 = nn.Conv2d(1024, num_classes,kernel_size=1)
        self.out2 = nn.Conv2d(512, num_classes,kernel_size=1)
        self.out = nn.Conv2d(128, num_classes,kernel_size=1)

    def forward(self, x):

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1 = self.encoder1(e0)

        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        x1 = self.swin.features[0](x)
        x1 = self.swin.features[1](x1)
        x2 = self.swin.features[2](x1)
        x2 = self.swin.features[3](x2)

        x3 = self.swin.features[4](x2)
        x3 = self.swin.features[5](x3)
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)
        x3 = x3.permute(0,3,1,2)
        
        all1 = torch.cat([x1, e2], dim=1)
        all2 = torch.cat([x2, e3], dim=1)
        all3 = torch.cat([x3, e4], dim=1)
        
        all1 = self.CA_1(all1)     #通道注意力 
        all2 = self.CA_2(all2) 
        all3 = self.CA_3(all3) 
 

        cdc1 = self.cdc_1(all1)  
        cdc2 = self.cdc_2(all2)      
        cdc3 = self.cdc_3(all3)       #交叉膨胀卷积 CDC PD

        all4 = self.hfa1(cdc3, cdc2, cdc1)   #并行编码器
        out4 = all4 
        
        

        all4 = self.down(all4)
        all4 = self.conv(all4)


        d3 = self.PG1(all3, all4) #注意力融合AF
        d3 = d3 + all4
        d3 = self.decoder3(d3)
        out3 = d3
        d3 = self.up3_1(d3)
       

        d2 = self.PG3(d3, all2)    
        d2 = d2 + all2
        d2 = self.decoder2(d2)
        out2 = d2
        d2 = self.up2_1(d2)

        d1 = self.PG2(d2 , all1)
        d1 = d1 + all1
        d1 = self.decoder1(d1)
        d1 = self.up1_1(d1)

        out4 = self.out4(out4)
        out4 = F.interpolate(out4, scale_factor=4, mode='bilinear') 

        out3 = self.out3(out3)
        out3 = F.interpolate(out3, scale_factor=16, mode='bilinear') 

        out2 = self.out2(out2)
        out2 = F.interpolate(out2, scale_factor=8, mode='bilinear') 
        
        out = self.out(d1)

        return out4, out3, out2, out

if __name__ == '__main__':

    x = torch.rand(5, 3, 224, 224).cuda()
    model = ParaTransCNN().cuda()
    y = model(x)




