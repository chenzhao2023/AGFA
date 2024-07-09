from collections import OrderedDict
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)

def deconv(in_channels, out_channels):  # This is upsample
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

def split_channel(tensor, num_chunks):  
    chunks = torch.chunk(tensor, num_chunks, dim=1)
    # chunks = split_channel(tensor, num_chunks)
    tensor1=tensor
    tensor2=tensor
    tensor3=tensor
    tensor4=tensor
    tensor1,tensor2,tensor3,tensor4=chunks
    return tensor1,tensor2,tensor3,tensor4

class Split_ChannelFeature_Comb(nn.Module):
    def __init__(self):
        super(Split_ChannelFeature_Comb, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, dilation=4, padding=4)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, tensor):
        chunks = torch.chunk(tensor, 4, dim=1)
        
        out1 = self.sigmoid(self.conv1(chunks[0]))  # [1,64,8,10,10]
        out2 = self.sigmoid(self.conv2(chunks[1]))
        out3 = self.sigmoid(self.conv3(chunks[2]))
        out4 = self.sigmoid(self.conv4(chunks[3]))
        merged_tensor = torch.cat((torch.mul(out1 , chunks[0]), torch.mul(out2 , chunks[1]), torch.mul(out3 , chunks[2]), torch.mul(out4 , chunks[3])), dim=1)

        return merged_tensor    # [1,128,8,10,10]
        

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ChannelAttention2(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention2, self).__init__()
        self.avg_pool= nn.AdaptiveAvgPool3d(1)
        self.max_pool= nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 16, kernel_size=1,bias=False)
        self.relu=nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes//16, in_planes, kernel_size=1,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):  #  x.shape:   [1, 32, 32, 40, 40])
        avg_out=self.avg_pool(x)
        avg_out=self.fc1(avg_out)
        avg_out=self.relu(avg_out)
        avg_out=self.fc2(avg_out)
        
        max_out=self.max_pool(x)
        max_out=self.fc1(max_out)
        max_out=self.relu(max_out)
        max_out=self.fc2(max_out)
        
        out=avg_out+ max_out
        out=self.sigmoid(out)
        return torch.mul(out , x)

class SpatialAttention2(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention2, self).__init__()
        self.conv1=nn.Conv3d(2,1, kernel_size=kernel_size, padding=3, bias=False)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):  # x: [1, 32, 32, 40, 40])
        avg_out = torch.mean(x, dim=1,keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out=torch.cat([avg_out,max_out], dim=1)
        out=self.conv1(out)
        out=self.sigmoid(out)
        return torch.mul(out , x)

class combineAttention(nn.Module):
    def __init__(self, in_channels):
        super(combineAttention,self).__init__()
        self.channel_attention=ChannelAttention2(in_channels)
        self.spatial_attention=SpatialAttention2(kernel_size=7)
        
    def forward(self, inputs):
        x=self.channel_attention(inputs)
        x=self.spatial_attention(x)
        return x

class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        #out += residual
        residual2 = torch.add(residual, out)
        out = self.relu(residual2)
        return out
    
class ResEncoder3d_Dila(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d_Dila, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)  # padding由 1 变成 2 
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):   # [1, 64, 16, 20, 20])
        residual = self.conv1x1(x)  # [1, 128, 16, 20, 20])
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))  # 1, [1, 128, 16, 20, 20])
        #out += residual
        residual2 = torch.add(residual, out)
        out = self.relu(residual2)
        return out

class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: slice number (depth)
        """
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,H*W*D,C]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,C,H*W*D]

        affinity1 = torch.matmul(proj_query, proj_key)
        affinity2 = torch.matmul(proj_judge, proj_key)
        affinity = torch.matmul(affinity1, affinity2)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out

class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxD )
        :return: affinity value + x
        """
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out

class AffinityAttention3d(nn.Module):    # 
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab + x
        return out


class AGFANet(nn.Module):
    def __init__(self, classes=3, channels=1):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(AGFANet, self).__init__()
        self.enc_input = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        # self.encoder3 = ResEncoder3d_Dila(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.combAttention1=combineAttention(32)
        self.combAttention2=combineAttention(64)
        self.combAttention3=combineAttention(128)
        self.split_channelFeature_comb=Split_ChannelFeature_Comb()
        self.affinity_attention = AffinityAttention3d(256)
        self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        self.decoder4 = Decoder3d(256, 128)
        self.decoder3 = Decoder3d(128, 64)
        self.decoder2 = Decoder3d(64, 32)
        self.decoder1 = Decoder3d(32, 16)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        self.sigmoid=nn.Sigmoid()
        self.final = nn.Conv3d(16, classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)   
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)
 
        input_feature = self.encoder4(down4)  #  input_feature:  [1, 256, 8, 10, 10])

        # Do decoder operations here
        up4 = self.deconv4(input_feature)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        # print("up3 : ", up3)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)        # [1, 2, 128, 160, 160])
        final = self.sigmoid(final)        # [1, 2, 128, 160, 160])
        return final
        

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
