"""
backend/model.py

U-Net implementation for binary image segmentation / background removal.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d -> BatchNorm2d -> ReLU blocks.

    This is the basic building block used in the U-Net encoder and decoder.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then DoubleConv.

    If bilinear is True, use nn.Upsample + 1x1 conv.
    Otherwise, use ConvTranspose2d for upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.bilinear = bilinear

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Pad x1 to match x2 size (in case of rounding issues)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to produce logits for each class."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for binary segmentation.

    Args:
        n_channels: Number of input channels (3 for RGB images).
        n_classes: Number of output channels (1 for binary mask).
        bilinear: Use bilinear upsampling instead of transposed conv.
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True, base_channels: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            Logits tensor of shape (B, n_classes, H, W).
            Apply sigmoid() during evaluation for probabilities.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _initialize_weights(self) -> None:
        """Initialize convolutional layers with Kaiming normal weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


def build_unet(
    n_channels: int = 3,
    n_classes: int = 1,
    bilinear: bool = True,
    base_channels: int = 64,
) -> UNet:
    """
    Helper to build a classic U-Net model.

    This keeps backward compatibility with earlier experiments.
    """
    return UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        base_channels=base_channels,
    )


# ---------------------------------------------------------------------------
# U²-Net style architecture (U2NET)
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """
    Simple Conv2d -> BatchNorm2d -> ReLU block with optional dilation.

    This mirrors the `conv_block` used in the provided TensorFlow reference.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RSUL(nn.Module):
    """
    Residual U-Block with pooling (generalized RSU_L from the TF reference).

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count (also used for the residual connection).
        mid_channels: Intermediate channels used inside the block.
        num_layers:   Depth of the internal U-Net structure (>= 3).
        bridge_dilation: Dilation rate for the bridge conv (corresponds to `rate` in TF).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        num_layers: int,
        bridge_dilation: int = 2,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("RSUL num_layers must be >= 3")

        self.num_layers = num_layers

        # Initial conv: adapt input to out_channels, used for residual addition.
        self.initial = ConvBlock(in_channels, out_channels, dilation=1)

        # Encoder
        self.enc_conv0 = ConvBlock(out_channels, mid_channels, dilation=1)
        self.enc_pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(num_layers - 2)]
        )
        self.enc_convs = nn.ModuleList(
            [ConvBlock(mid_channels, mid_channels, dilation=1) for _ in range(num_layers - 2)]
        )

        # Bridge (dilated conv)
        self.bridge = ConvBlock(mid_channels, mid_channels, dilation=bridge_dilation)

        # Decoder
        # First decoder conv after concatenating bridge with the first skip.
        self.dec_conv0 = ConvBlock(mid_channels * 2, mid_channels, dilation=1)
        # Additional decoder convs for intermediate upsampling steps.
        self.dec_convs = nn.ModuleList(
            [ConvBlock(mid_channels * 2, mid_channels, dilation=1) for _ in range(num_layers - 3)]
        )
        # Final conv to project back to out_channels.
        self.final_conv = ConvBlock(mid_channels * 2, out_channels, dilation=1)

        # Upsamplers for decoder (number of upsample steps = num_layers - 2)
        self.ups = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                for _ in range(num_layers - 2)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv + residual feature
        x0 = self.initial(x)
        init_feats = x0

        # Encoder
        skips: list[torch.Tensor] = []
        x = self.enc_conv0(x0)
        skips.append(x)
        for pool, enc_conv in zip(self.enc_pools, self.enc_convs):
            x = pool(x)
            x = enc_conv(x)
            skips.append(x)

        # Bridge
        x = self.bridge(x)

        # Decoder
        skips_rev = list(reversed(skips))

        # First decoder conv (no upsample before this step)
        x = torch.cat([x, skips_rev[0]], dim=1)
        x = self.dec_conv0(x)

        # Intermediate upsample + conv steps
        for i in range(self.num_layers - 3):
            x = self.ups[i](x)
            x = torch.cat([x, skips_rev[i + 1]], dim=1)
            x = self.dec_convs[i](x)

        # Final upsample and projection to out_channels
        x = self.ups[-1](x)
        x = torch.cat([x, skips_rev[-1]], dim=1)
        x = self.final_conv(x)

        # Residual addition
        return x + init_feats


class RSU4F(nn.Module):
    """
    RSU-4F block from U²-Net (multi-scale dilated convolutions without pooling).

    This corresponds to RSU_4F in the TensorFlow reference.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.conv_in = ConvBlock(in_channels, out_channels, dilation=1)

        # Encoder (dilated)
        self.conv1 = ConvBlock(out_channels, mid_channels, dilation=1)
        self.conv2 = ConvBlock(mid_channels, mid_channels, dilation=2)
        self.conv3 = ConvBlock(mid_channels, mid_channels, dilation=4)

        # Bridge
        self.conv4 = ConvBlock(mid_channels, mid_channels, dilation=8)

        # Decoder
        self.conv3d = ConvBlock(mid_channels * 2, mid_channels, dilation=4)
        self.conv2d = ConvBlock(mid_channels * 2, mid_channels, dilation=2)
        self.conv1d = ConvBlock(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_in(x)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = self.conv4(x3)

        x = torch.cat([x4, x3], dim=1)
        x = self.conv3d(x)

        x = torch.cat([x, x2], dim=1)
        x = self.conv2d(x)

        x = torch.cat([x, x1], dim=1)
        x = self.conv1d(x)

        return x + x0


class U2NET(nn.Module):
    """
    U²-Net style encoder-decoder using RSU blocks.

    This is an adaptation of the provided TensorFlow U²-Net reference to PyTorch,
    simplified to output a single segmentation logit map (no public side outputs)
    so that it plugs into the existing training and inference pipeline.

    The input/output conventions match the existing UNet:

        - Input : (B, n_channels, H, W)
        - Output: (B, num_classes, H, W) logits
    """

    def __init__(
        self,
        n_channels: int = 3,
        num_classes: int = 1,
        out_ch: list[int] | None = None,
        int_ch: list[int] | None = None,
    ):
        super().__init__()

        if out_ch is None:
            out_ch = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
        if int_ch is None:
            int_ch = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]

        if len(out_ch) != 11 or len(int_ch) != 11:
            raise ValueError("out_ch and int_ch must have length 11.")

        # Encoder
        self.stage1 = RSUL(n_channels, out_ch[0], int_ch[0], num_layers=7, bridge_dilation=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = RSUL(out_ch[0], out_ch[1], int_ch[1], num_layers=6, bridge_dilation=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = RSUL(out_ch[1], out_ch[2], int_ch[2], num_layers=5, bridge_dilation=2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.stage4 = RSUL(out_ch[2], out_ch[3], int_ch[3], num_layers=4, bridge_dilation=2)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.stage5 = RSU4F(out_ch[3], out_ch[4], int_ch[4])
        self.pool5 = nn.MaxPool2d(2, 2)

        # Bridge
        self.bridge = RSU4F(out_ch[4], out_ch[5], int_ch[5])

        # Decoder
        self.stage6d = RSU4F(out_ch[5] + out_ch[4], out_ch[6], int_ch[6])
        self.stage5d = RSUL(out_ch[6] + out_ch[3], out_ch[7], int_ch[7], num_layers=4, bridge_dilation=2)
        self.stage4d = RSUL(out_ch[7] + out_ch[2], out_ch[8], int_ch[8], num_layers=5, bridge_dilation=2)
        self.stage3d = RSUL(out_ch[8] + out_ch[1], out_ch[9], int_ch[9], num_layers=6, bridge_dilation=2)
        self.stage2d = RSUL(out_ch[9] + out_ch[0], out_ch[10], int_ch[10], num_layers=7, bridge_dilation=2)

        # Final projection to segmentation logits.
        # We keep a single output head to remain compatible with the current pipeline.
        self.out_conv = nn.Conv2d(out_ch[10], num_classes, kernel_size=1)

    def _upsample_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Upsample `x` to match the spatial size of `ref` using bilinear interpolation."""
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.stage1(x)
        p1 = self.pool1(s1)

        s2 = self.stage2(p1)
        p2 = self.pool2(s2)

        s3 = self.stage3(p2)
        p3 = self.pool3(s3)

        s4 = self.stage4(p3)
        p4 = self.pool4(s4)

        s5 = self.stage5(p4)
        p5 = self.pool5(s5)

        # Bridge
        b1 = self.bridge(p5)
        b2 = F.interpolate(b1, scale_factor=2, mode="bilinear", align_corners=True)

        # Decoder
        d1 = self.stage6d(torch.cat([b2, s5], dim=1))
        u1 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)

        d2 = self.stage5d(torch.cat([u1, s4], dim=1))
        u2 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)

        d3 = self.stage4d(torch.cat([u2, s3], dim=1))
        u3 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)

        d4 = self.stage3d(torch.cat([u3, s2], dim=1))
        u4 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)

        d5 = self.stage2d(torch.cat([u4, s1], dim=1))

        # Single segmentation head (logits)
        logits = self.out_conv(d5)
        return logits


def build_u2net(
    n_channels: int = 3,
    n_classes: int = 1,
    out_ch: list[int] | None = None,
    int_ch: list[int] | None = None,
) -> U2NET:
    """
    Build a full U²-Net model (higher capacity).

    By default this uses the channel configuration from the provided reference.
    """
    return U2NET(
        n_channels=n_channels,
        num_classes=n_classes,
        out_ch=out_ch,
        int_ch=int_ch,
    )


def build_u2net_lite(
    n_channels: int = 3,
    n_classes: int = 1,
) -> U2NET:
    """
    Build a lighter U²-Net variant suitable for limited GPU memory.

    This mirrors the `build_u2net_lite` configuration from the TensorFlow example.
    """
    out_ch = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    int_ch = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    return U2NET(
        n_channels=n_channels,
        num_classes=n_classes,
        out_ch=out_ch,
        int_ch=int_ch,
    )


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test the lighter U²-Net variant on a 512x512 RGB image.
    model = build_u2net(n_channels=3, n_classes=1).to(device)
    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)

    print("U2NET-lite sanity check")
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)