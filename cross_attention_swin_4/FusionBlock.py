from numpy import dtype

from cross_attention_swin_4.model import SwinTransformer_2
from cross_attention_swin_4.cross_attention import CrossAttentionBlock1, CrossAttentionBlock
from cross_attention_swin_4.cross_attention import CrossAttentionBlock2
from cross_attention_swin_4.swin_transformer import SwinTransformer
from cross_attention_swin_4.model2 import MobileViT
from cross_attention_swin_4.model_config import get_config
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# import sys
# class Logger(object):
#     def __init__(self, filename='test.log', stream=sys.stdout):
#         self.terminal = stream
#         self.log = open(filename, 'w')
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
#
# # 将控制台的结果输出到a.log文件，可以改成a.txt
# sys.stdout = Logger('test.log', sys.stdout)
# sys.stderr = Logger('test.log_file', sys.stderr)
#


class Fusioncls(nn.Module):
    def __init__(self,):
        super(Fusioncls, self).__init__()
        self.model1 = SwinTransformer_2(in_chans=3,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=1)
        self.model2 = MobileViT(get_config("x_small"), num_classes=1)
        self.model3 = SwinTransformer(in_chans=1,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=1)
        self.cross_attn1 = CrossAttentionBlock1(dim=768, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.cross_attn2 = CrossAttentionBlock2(dim=384, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.cross_attn3 = CrossAttentionBlock(dim=1152, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.projs1 = nn.Sequential(
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 384),
        )
        self.projs2 = nn.Sequential(
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 768),
        )
        # self.jiaquan = nn.Sequential()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1152, 1)
        # self.rtt = reshape_to_target(x=input, target_shape=(36, 1536, 1))
        # self.rrr = ImageTransformNet()



    # def forward(self, x1, x2):
    def forward(self, inputs):

        inputs_1 = inputs[:, 0, :, :]
        inputs_1 = inputs_1.unsqueeze(1)
        inputs_2 = inputs[:, 1, :, :]
        inputs_2 = inputs_2.unsqueeze(1)
        inputs_3 = inputs[:, 2, :, :]
        inputs_3 = inputs_3.unsqueeze(1)
        x1 = torch.cat((inputs_1, inputs_2, inputs_3), 1) #[2, 3, 128, 128]
        A1 = x1
        A1 = self.pool(F.relu(self.conv1(A1)))  # 大小变为: [batch_size, 16, 64, 64]
        A1 = self.pool(F.relu(self.conv2(A1)))  # 大小变为: [batch_size, 32, 32, 32]
        A1 = self.pool(F.relu(self.conv3(A1)))  # 大小变为: [batch_size, 1, 16, 16]

        # 为了达到 [48, 48, 1]，可以在最后添加一个上采样层
        A1 = F.interpolate(A1, size=(48, 48), mode='bilinear', align_corners=False)  # 大小变为: [batch_size, 1, 48, 48]

        # 去除多余的维度 [batch_size, 1, 48, 48] -> [batch_size, 48, 48]
        A1 = torch.squeeze(A1, dim=1)

        inputs_4 = inputs[:, 3, :, :]
        inputs_4 = inputs_4.unsqueeze(1)
        inputs_5 = inputs[:, 3, :, :]
        inputs_5 = inputs_5.unsqueeze(1)
        inputs_6 = inputs[:, 3, :, :]
        inputs_6 = inputs_6.unsqueeze(1)
        x2 = torch.cat((inputs_4, inputs_5, inputs_6), 1)   #[2, 3, 128, 128]
        B1 = x2
        B1 = self.pool(F.relu(self.conv1(B1)))  # 大小变为: [batch_size, 16, 64, 64]
        B1 = self.pool(F.relu(self.conv2(B1)))  # 大小变为: [batch_size, 32, 32, 32]
        B1 = self.pool(F.relu(self.conv3(B1)))  # 大小变为: [batch_size, 1, 16, 16]

        # 为了达到 [48, 48, 1]，可以在最后添加一个上采样层
        B1 = F.interpolate(B1, size=(48, 48), mode='bilinear', align_corners=False)  # 大小变为: [batch_size, 1, 48, 48]

        # 去除多余的维度 [batch_size, 1, 48, 48] -> [batch_size, 48, 48]
        B1 = torch.squeeze(B1, dim=1)

        x1 = self.model1(x1)        #[36,1,768]
        x2 = self.model2(x2)        #[36,1,384]
        tokens = []
        tokens.append(x1)
        tokens.append(x2)

        x11 = self.projs1(x1)       #[36,1,384]
        x22 = self.projs2(x2)       #[36,1,768]

        cls_proj =[]
        cls_proj.append(x11)
        cls_proj.append(x22)

        fusion1 = torch.cat((tokens[0], cls_proj[1][:, 1:, ...]), dim=1)        #[36, 1, 768]
        fusion1 = self.cross_attn1(fusion1)                         #[36, 1, 768]
        fusion2 = torch.cat((tokens[1], cls_proj[0][:, 1:, ...]), dim=1)        #[36, 1, 384]
        # print('fusion2', fusion2.shape)
        fusion2 = self.cross_attn2(fusion2)                         #[36, 1, 384]
        # print('cross_attention(fusion)', fusion2.shape)

        fusion3 = torch.cat((tokens[0][:, 1:, ...], cls_proj[1]), dim=1)        #[36, 1, 768] [B,L,C]
        fusion3 = self.cross_attn1(fusion3)
        fusion4 = torch.cat((tokens[1][:, 1:, ...], cls_proj[0]), dim=1)        #[36, 1, 384] [B,L,C]
        fusion4 = self.cross_attn2(fusion4)

        A = torch.cat((fusion1, fusion2), dim=2) #[36, 1, 1152]
        # print('A', A.shape)
        A = A.reshape(A.shape[0], 24, 48)
        # print('A.reshape', A.shape)
        attn1 = (A1  @ A.transpose(-2, -1)) #[36,48,24]
        # print('attn1', attn1.shape)
        attn1 = torch.add(A.transpose(1,2), attn1)
        # print('add_attn1', attn1.shape)
        attn2 = (B1 @ A.transpose(-2, -1))  #[36,48,24]
        # print('attn2', attn2.shape)
        attn2 = torch.add(A.transpose(1,2), attn2) #[36,48,24]
        # print('add_attn2', attn2.shape)
        atten3 = (attn1  @ attn2.transpose(-2, -1))
        A1B1 = torch.add(A1, B1)
        atten3A1B1 = (atten3  @ A1B1)
        # print('atten3A1B1', atten3A1B1.shape)
        atten1 =(atten3A1B1  @ attn1.transpose(1,2).transpose(-2, -1))
        # print('XXXXXX', atten1.shape)
        atten2 = (atten3A1B1 @ attn2.transpose(1, 2).transpose(-2, -1))
        # print('YYYYYY', atten1.shape)
        attn1 = atten1.reshape(atten1.shape[0], 1, 1152)
        # print('attn1.reshape', attn1.shape)
        attn2 = atten2.reshape(atten2.shape[0], 1, 1152)
        # print('attn2.reshape', attn2.shape)
        attn1 = self.cross_attn3(attn1)
        # print('cross_attn1', attn1.shape)
        attn2 = self.cross_attn3(attn2)
        # print('cross_attn2', attn2.shape)
        x = torch.cat((attn1, attn2), dim=2)
        # print('x_connect', x.shape)


        # x = torch.cat((fusion1, fusion2, fusion3, fusion4), dim=2)          # [36, 1, 2304] [B,L,C]
        # print('1', x.shape)
        # x = torch.cat((fusion3, fusion4), dim=2)
        # x = self.avgpool(x.transpose(1, 2))
        x = x.transpose(1, 2)                           #[36, 2304, 1] [B,C,L]
        # x = self.rtt(x)                              #[36, 1, 1] [B,C,L]
        # x = x.reshape_to_target(x)
        # print('x.transpose', x.shape)
        # target_shape = (36, 1536, 1)
        # x = reshape_to_target(x, target_shape)
        # print('XXXX', x.shape)
        # print(x.shape)
        # x = x.view(8, 54, 64, 2)
        # x = x.resize(36, 1536, 1)
        # x = x.reshape(1, 4, 128, 108)
        x = x.reshape(x.shape[0], 1, 48, 48)
        # print('3', x.shape)
        x = self.model3(x)                                #[1, 768, 1] [B,C,L]
        # print('7', x.shape)
        # x = torch.flatten(x, 1)
        # print('5', x.shape)
        # x = self.head(x)
        # print('6', x.shape)
        return x


         #
        # x = torch.flatten(x, 1)  # [8,1152]
        # x = self.head(x)

        # return x


# input_shape = [128, 128]  # 设置输入大小
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择是否使用GPU
# model = Fusioncls().to(device)  # 实例化网络
# summary(model, (4, input_shape[0], input_shape[1]))

'''
x1 torch.Size([2, 3, 128, 128])
A1 torch.Size([2, 48, 48])
A1.dtype torch.float32
x2 torch.Size([2, 3, 128, 128])
B1 torch.Size([2, 48, 48])
B1.dtype torch.float32
model1(X1) torch.Size([2, 1, 768])
model2(X2) torch.Size([2, 1, 384])
fusion1 torch.Size([2, 1, 768])
cross_attention(fusion) torch.Size([2, 1, 768])
fusion2 torch.Size([2, 1, 384])
cross_attention(fusion) torch.Size([2, 1, 384])
A torch.Size([2, 1, 1152])
A.reshape torch.Size([2, 24, 48])
attn1 torch.Size([2, 48, 24])
add_attn1 torch.Size([2, 48, 24])
attn2 torch.Size([2, 48, 24])
add_attn2 torch.Size([2, 48, 24])
attn1.reshape torch.Size([2, 1, 1152])
attn2.reshape torch.Size([2, 1, 1152])
cross_attn1 torch.Size([2, 1, 1152])
cross_attn2 torch.Size([2, 1, 1152])
x_connect torch.Size([2, 1, 2304])
x.transpose torch.Size([2, 2304, 1])
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 16, 128, 128]        448
├─MaxPool2d: 1-2                         [-1, 16, 64, 64]          --
├─Conv2d: 1-3                            [-1, 32, 64, 64]          4,640
├─MaxPool2d: 1-4                         [-1, 32, 32, 32]          --
├─Conv2d: 1-5                            [-1, 1, 32, 32]           289
├─MaxPool2d: 1-6                         [-1, 1, 16, 16]           --
├─Conv2d: 1-7                            [-1, 16, 128, 128]        (recursive)
├─MaxPool2d: 1-8                         [-1, 16, 64, 64]          --
├─Conv2d: 1-9                            [-1, 32, 64, 64]          (recursive)
├─MaxPool2d: 1-10                        [-1, 32, 32, 32]          --
├─Conv2d: 1-11                           [-1, 1, 32, 32]           (recursive)
├─MaxPool2d: 1-12                        [-1, 1, 16, 16]           --
├─SwinTransformer_2: 1-13                [-1, 1, 768]              --
|    └─PatchEmbed: 2-1                   [-1, 1024, 96]            --
|    |    └─Conv2d: 3-1                  [-1, 96, 32, 32]          4,704
|    |    └─LayerNorm: 3-2               [-1, 1024, 96]            192
|    └─Dropout: 2-2                      [-1, 1024, 96]            --
|    └─ModuleList: 2                     []                        --
|    |    └─BasicLayer: 3-3              [-1, 256, 192]            299,190
|    |    └─BasicLayer: 3-4              [-1, 64, 384]             1,188,204
|    |    └─BasicLayer: 3-5              [-1, 16, 768]             11,841,672
|    |    └─BasicLayer: 3-6              [-1, 16, 768]             14,183,856
|    └─LayerNorm: 2-3                    [-1, 16, 768]             1,536
|    └─AdaptiveAvgPool1d: 2-4            [-1, 768, 1]              --
├─MobileViT: 1-14                        [-1, 1, 384]              --
|    └─ConvLayer: 2-5                    [-1, 16, 64, 64]          --
|    |    └─Sequential: 3-7              [-1, 16, 64, 64]          464
|    └─Sequential: 2-6                   [-1, 32, 64, 64]          --
|    |    └─InvertedResidual: 3-8        [-1, 32, 64, 64]          3,968
|    └─Sequential: 2-7                   [-1, 48, 32, 32]          --
|    |    └─InvertedResidual: 3-9        [-1, 48, 32, 32]          12,000
|    |    └─InvertedResidual: 3-10       [-1, 48, 32, 32]          21,024
|    |    └─InvertedResidual: 3-11       [-1, 48, 32, 32]          21,024
|    └─Sequential: 2-8                   [-1, 64, 16, 16]          --
|    |    └─InvertedResidual: 3-12       [-1, 64, 16, 16]          24,128
|    |    └─MobileViTBlock: 3-13         [-1, 64, 16, 16]          273,024
|    └─Sequential: 2-9                   [-1, 80, 8, 8]            --
|    |    └─InvertedResidual: 3-14       [-1, 80, 8, 8]            40,352
|    |    └─MobileViTBlock: 3-15         [-1, 80, 8, 8]            658,800
|    └─Sequential: 2-10                  [-1, 96, 4, 4]            --
|    |    └─InvertedResidual: 3-16       [-1, 96, 4, 4]            60,672
|    |    └─MobileViTBlock: 3-17         [-1, 96, 4, 4]            779,760
|    └─ConvLayer: 2-11                   [-1, 384, 4, 4]           --
|    |    └─Sequential: 3-18             [-1, 384, 4, 4]           37,632
|    └─AdaptiveAvgPool1d: 2-12           [-1, 384, 1]              --
├─Sequential: 1-15                       [-1, 1, 384]              --
|    └─LayerNorm: 2-13                   [-1, 1, 768]              1,536
|    └─GELU: 2-14                        [-1, 1, 768]              --
|    └─Linear: 2-15                      [-1, 1, 384]              295,296
├─Sequential: 1-16                       [-1, 1, 768]              --
|    └─LayerNorm: 2-16                   [-1, 1, 384]              768
|    └─GELU: 2-17                        [-1, 1, 384]              --
|    └─Linear: 2-18                      [-1, 1, 768]              295,680
├─CrossAttentionBlock1: 1-17             [-1, 1, 768]              --
|    └─LayerNorm: 2-19                   [-1, 1, 768]              1,536
|    └─CrossAttention1: 2-20             [-1, 1, 768]              --
|    |    └─Linear: 3-19                 [-1, 1, 768]              590,592
|    |    └─Linear: 3-20                 [-1, 1, 768]              590,592
|    |    └─Linear: 3-21                 [-1, 1, 768]              590,592
|    |    └─Dropout: 3-22                [-1, 8, 1, 1]             --
|    |    └─Linear: 3-23                 [-1, 1, 768]              590,592
|    |    └─Dropout: 3-24                [-1, 1, 768]              --
|    └─Identity: 2-21                    [-1, 1, 768]              --
├─CrossAttentionBlock2: 1-18             [-1, 1, 384]              --
|    └─LayerNorm: 2-22                   [-1, 1, 384]              768
|    └─CrossAttention2: 2-23             [-1, 1, 384]              --
|    |    └─Linear: 3-25                 [-1, 1, 384]              147,840
|    |    └─Linear: 3-26                 [-1, 1, 384]              147,840
|    |    └─Linear: 3-27                 [-1, 1, 384]              147,840
|    |    └─Dropout: 3-28                [-1, 8, 1, 1]             --
|    |    └─Linear: 3-29                 [-1, 1, 384]              147,840
|    |    └─Dropout: 3-30                [-1, 1, 384]              --
|    └─Identity: 2-24                    [-1, 1, 384]              --
├─CrossAttentionBlock1: 1-19             [-1, 1, 768]              (recursive)
|    └─LayerNorm: 2-25                   [-1, 1, 768]              (recursive)
|    └─CrossAttention1: 2-26             [-1, 1, 768]              (recursive)
|    |    └─Linear: 3-31                 [-1, 1, 768]              (recursive)
|    |    └─Linear: 3-32                 [-1, 1, 768]              (recursive)
|    |    └─Linear: 3-33                 [-1, 1, 768]              (recursive)
|    |    └─Dropout: 3-34                [-1, 8, 1, 1]             --
|    |    └─Linear: 3-35                 [-1, 1, 768]              (recursive)
|    |    └─Dropout: 3-36                [-1, 1, 768]              --
|    └─Identity: 2-27                    [-1, 1, 768]              --
├─CrossAttentionBlock2: 1-20             [-1, 1, 384]              (recursive)
|    └─LayerNorm: 2-28                   [-1, 1, 384]              (recursive)
|    └─CrossAttention2: 2-29             [-1, 1, 384]              (recursive)
|    |    └─Linear: 3-37                 [-1, 1, 384]              (recursive)
|    |    └─Linear: 3-38                 [-1, 1, 384]              (recursive)
|    |    └─Linear: 3-39                 [-1, 1, 384]              (recursive)
|    |    └─Dropout: 3-40                [-1, 8, 1, 1]             --
|    |    └─Linear: 3-41                 [-1, 1, 384]              (recursive)
|    |    └─Dropout: 3-42                [-1, 1, 384]              --
|    └─Identity: 2-30                    [-1, 1, 384]              --
├─CrossAttentionBlock: 1-21              [-1, 1, 1152]             --
|    └─LayerNorm: 2-31                   [-1, 1, 1152]             2,304
|    └─CrossAttention: 2-32              [-1, 1, 1152]             --
|    |    └─Linear: 3-43                 [-1, 1, 1152]             1,328,256
|    |    └─Linear: 3-44                 [-1, 1, 1152]             1,328,256
|    |    └─Linear: 3-45                 [-1, 1, 1152]             1,328,256
|    |    └─Dropout: 3-46                [-1, 8, 1, 1]             --
|    |    └─Linear: 3-47                 [-1, 1, 1152]             1,328,256
|    |    └─Dropout: 3-48                [-1, 1, 1152]             --
|    └─Identity: 2-33                    [-1, 1, 1152]             --
├─CrossAttentionBlock: 1-22              [-1, 1, 1152]             (recursive)
|    └─LayerNorm: 2-34                   [-1, 1, 1152]             (recursive)
|    └─CrossAttention: 2-35              [-1, 1, 1152]             (recursive)
|    |    └─Linear: 3-49                 [-1, 1, 1152]             (recursive)
|    |    └─Linear: 3-50                 [-1, 1, 1152]             (recursive)
|    |    └─Linear: 3-51                 [-1, 1, 1152]             (recursive)
|    |    └─Dropout: 3-52                [-1, 8, 1, 1]             --
|    |    └─Linear: 3-53                 [-1, 1, 1152]             (recursive)
|    |    └─Dropout: 3-54                [-1, 1, 1152]             --
|    └─Identity: 2-36                    [-1, 1, 1152]             --
├─SwinTransformer: 1-23                  [-1, 1]                   --
|    └─PatchEmbed: 2-37                  [-1, 144, 96]             --
|    |    └─Conv2d: 3-55                 [-1, 96, 12, 12]          1,632
|    |    └─LayerNorm: 3-56              [-1, 144, 96]             192
|    └─Dropout: 2-38                     [-1, 144, 96]             --
|    └─ModuleList: 2                     []                        --
|    |    └─BasicLayer: 3-57             [-1, 36, 192]             299,190
|    |    └─BasicLayer: 3-58             [-1, 9, 384]              1,188,204
|    |    └─BasicLayer: 3-59             [-1, 4, 768]              11,841,672
|    |    └─BasicLayer: 3-60             [-1, 4, 768]              14,183,856
|    └─LayerNorm: 2-39                   [-1, 4, 768]              1,536
|    └─AdaptiveAvgPool1d: 2-40           [-1, 768, 1]              --
|    └─Linear: 2-41                      [-1, 1]                   769
==========================================================================================
Total params: 65,839,270
Trainable params: 65,839,270
Non-trainable params: 0
Total mult-adds (M): 229.93
==========================================================================================
Input size (MB): 0.25
Forward/backward pass size (MB): 6.04
Params size (MB): 251.16
Estimated Total Size (MB): 257.44
==========================================================================================

'''