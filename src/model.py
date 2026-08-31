import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CONVOLUTIONAL BLOCK
# ============================================================

class ConvBlock(nn.Module):
    """
    Two convolution layers followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):
    """
    Upsampling decoder block with skip connection.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        # Upsample decoder feature map to the
        # spatial resolution of the skip connection.

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Concatenate decoder features and skip features.

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.conv(x)


# ============================================================
# RESNET18 U-NET
# ============================================================

class ResNet18UNet(nn.Module):
    """
    Baseline ResNet-18 U-Net.

    Input:
        [B, 3, H, W]

    Output:
        [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes=5,
        pretrained=True
    ):
        super().__init__()

        # ----------------------------------------------------
        # ResNet-18 backbone
        # ----------------------------------------------------

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet18(
            weights=weights
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.input_block = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=32
        )

        # ----------------------------------------------------
        # FINAL SEGMENTATION HEAD
        #
        # This is the important baseline structure.
        #
        # decoder1 output:
        #     32 channels
        #
        # final_conv:
        #     32 -> 32 -> 5
        #
        # This matches the baseline checkpoint.
        # ----------------------------------------------------

        self.final_conv = nn.Sequential(

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                num_classes,
                kernel_size=1,
                bias=True
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        x0 = self.input_block(x)

        x1 = self.maxpool(x0)
        x1 = self.encoder1(x1)

        x2 = self.encoder2(x1)

        x3 = self.encoder3(x2)

        x4 = self.encoder4(x3)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        d4 = self.decoder4(
            x4,
            x3
        )

        d3 = self.decoder3(
            d4,
            x2
        )

        d2 = self.decoder2(
            d3,
            x1
        )

        d1 = self.decoder1(
            d2,
            x0
        )

        # ----------------------------------------------------
        # Restore original input resolution
        # ----------------------------------------------------

        decoder_output = F.interpolate(
            d1,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Final segmentation prediction
        # ----------------------------------------------------

        output = self.final_conv(
            decoder_output
        )

        return output


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    num_classes=5,
    pretrained=True
):
    """
    Create the baseline ResNet18 U-Net.
    """

    model = ResNet18UNet(
        num_classes=num_classes,
        pretrained=pretrained
    )

    return model


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Baseline ResNet18 U-Net")
    print("=" * 70)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        num_classes=5,
        pretrained=False
    )

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        3,
        256,
        1024
    )

    print(
        f"Input shape: {x.shape}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(x)

    print(
        f"Output shape: {output.shape}"
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    expected_shape = (
        2,
        5,
        256,
        1024
    )

    assert output.shape == expected_shape, (
        f"Unexpected output shape: {output.shape}"
    )

    print()
    print("Model test PASSED")
    print("=" * 70)