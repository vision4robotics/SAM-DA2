import torch
import torch.nn as nn
from siamban.models.vmamba.vmamba import VMambaNeck, VMambaNeckV2
from siamban.models.GRL import GradientScalarLayer


class VMambaDiscriminator(nn.Module):
    def __init__(self, depth=1, channels=256, d_state=16, ssm_ratio=2.0, num_classes=1, patch_size=4, img_size=128, version='v1'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = channels
        self.patch_embed = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        if 'v1' in version.lower():
            self.block = VMambaNeck(depths=[1]*depth, dims=[channels]*depth)
        elif 'v2' in version.lower():
            self.block = VMambaNeckV2(depths=[1]*depth, dims=[channels]*depth)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.grl_img = GradientScalarLayer(-0.1)

    def forward(self, x):
        x = self.grl_img(x)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)  # b, h, w, c
        x = self.block(x)
        x = self.pool(x.permute(0, 3, 1, 2))
        x = self.head(x.flatten(1))
        return x

if __name__ == '__main__':
    x = torch.randn(1, 256, 128, 128).cuda()
    model = VMambaDiscriminator(channels=256,version='v2').cuda()
    total_params = sum(p.numel() for p in model.parameters())
    out = model(x)
    print(out.shape)    
    print(f"mamba discriminator have {total_params} parameters")