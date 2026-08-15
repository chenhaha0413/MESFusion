import torch
from torch import nn
from cross_attention_swin_4.model import SwinTransformer_2
from cross_attention_swin_4.cross_attention import CrossAttentionBlock1
from cross_attention_swin_4.cross_attention import CrossAttentionBlock2
from cross_attention_swin_4.swin_transformer import SwinTransformer
from cross_attention_swin_4.model2 import MobileViT
from cross_attention_swin_4.model_config import get_config
from torchsummary import summary

class Fusioncls2(nn.Module):
    def __init__(self, ):
        super(Fusioncls2, self).__init__()
        self.model1 = SwinTransformer_2(in_chans=3,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=1)
        # self.model2 = MobileViT(get_config("x_small"), num_classes=1)
        self.model2 = SwinTransformer_2(in_chans=3,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=1)
        self.model3 = SwinTransformer(in_chans=4,
                                      patch_size=4,
                                      window_size=7,
                                      embed_dim=96,
                                      depths=(2, 2, 6, 2),
                                      num_heads=(3, 6, 12, 24),
                                      num_classes=1)
        self.cross_attn1 = CrossAttentionBlock1(dim=768, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.cross_attn2 = CrossAttentionBlock2(dim=768, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.projs1 = nn.Sequential(
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 768),
        )
        self.projs2 = nn.Sequential(
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 768),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1152, 1)

    # def forward(self, x1, x2):
    def forward(self, inputs):

        inputs_1 = inputs[:, 0, :, :]
        inputs_1 = inputs_1.unsqueeze(1)
        inputs_2 = inputs[:, 1, :, :]
        inputs_2 = inputs_2.unsqueeze(1)
        inputs_3 = inputs[:, 2, :, :]
        inputs_3 = inputs_3.unsqueeze(1)
        x1 = torch.cat((inputs_1, inputs_2, inputs_3), 1)
        inputs_4 = inputs[:, 3, :, :]
        inputs_4 = inputs_4.unsqueeze(1)
        inputs_5 = inputs[:, 3, :, :]
        inputs_5 = inputs_5.unsqueeze(1)
        inputs_6 = inputs[:, 3, :, :]
        inputs_6 = inputs_6.unsqueeze(1)
        x2 = torch.cat((inputs_4, inputs_5, inputs_6), 1)

        x1 = self.model1(x1)  # [8,1,768]
        x2 = self.model2(x2)  # [8,1,768]

        tokens = []
        tokens.append(x1)
        tokens.append(x2)

        x11 = self.projs1(x1)  # [8,1,768]
        x22 = self.projs2(x2)  # [8,1,768]

        cls_proj = []
        cls_proj.append(x11)
        cls_proj.append(x22)

        fusion1 = torch.cat((tokens[0], cls_proj[1]), dim=1)  # [8,2,768]
        fusion1 = self.cross_attn1(fusion1)
        fusion2 = torch.cat((tokens[1], cls_proj[0]), dim=1)  # [8,2,768]
        fusion2 = self.cross_attn2(fusion2)

        # fusion3 = torch.cat((tokens[0][:, 1:, ...], cls_proj[1]), dim=1)        #[8,1,768] [B,L,C]
        # fusion3 = self.cross_attn1(fusion3)
        # fusion4 = torch.cat((tokens[1][:, 1:, ...], cls_proj[0]), dim=1)        #[8,1,384] [B,L,C]
        # fusion4 = self.cross_attn2(fusion4)

        x = torch.cat((fusion1, fusion2), dim=2)  # [8,2,768]+[8,2,384]=[8,2,1536] [B,L,C]
        print(x.shape)
        # x = torch.cat((fusion3, fusion4), dim=2)  # [8,1,384]+[8,1,768]=[8,1,1152]
        # x = self.avgpool(x.transpose(1, 2))  # [8,1536,1]
        x = x.transpose(1, 2)  # [8,1536,2]
        # print(x.shape)
        # x = x.view(8, 54, 64, 2)
        # i = 0
        # for i in range(epoch):
        #     if i == 1:
        #         i = i + 1
        #         x = x.view(1, 4, 128, 108)
        #         x = self.model3(x)
        #         return x
        #     else:
        #         i = i + 1
        #         x = x.view(1, 4, 64, 54)
        #         x = self.model3(x)
        #         return x
        x = x.view(1, 4, 64, 54)
        x = self.model3(x)
        return x
