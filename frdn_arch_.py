import torch
import torch.nn as nn
from rdn_arch import RDB

class FeedbackRDNBlock(nn.Module):
    def __init__(self, num_blocks, num_layers, num_features):
        super().__init__()
        self.rdbs = nn.Sequential(*[
            RDB(num_features, num_features, num_layers) for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.rdbs(x)

class FRDN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_blocks, num_layers, upscale_factor, T):
        super(FRDN, self).__init__()
        self.T = T
        self.upscale_factor = upscale_factor

        self.sfe1 = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.feedback_fuse = nn.Conv2d(2 * num_features, num_features, 1)

        self.refine = FeedbackRDNBlock(num_blocks, num_layers, num_features)

        r = upscale_factor
        if r in [2, 3]:
            self.upnet = nn.Sequential(
                nn.Conv2d(num_features, num_features * r * r, 3, padding=1),
                nn.PixelShuffle(r),
                nn.Conv2d(num_features, out_channels, 3, padding=1)
            )
        elif r == 4:
            self.upnet = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, 3, padding=1)
            )
        else:
            raise ValueError("Upscale factor must be 2, 3 or 4.")

    def forward(self, x):
        f1 = self.sfe1(x)
        F0 = self.sfe2(f1)

        hidden = torch.zeros_like(F0)

        upsampled_lr = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        outputs = []
        for t in range(self.T):
            fused_input = self.feedback_fuse(torch.cat([F0, hidden], dim=1))
            hidden = self.refine(fused_input)
            residual = self.upnet(hidden)
            sr = residual + upsampled_lr
            outputs.append(sr)

        return outputs
