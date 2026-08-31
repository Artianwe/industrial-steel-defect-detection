import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MULTICLASS DICE LOSS
# ============================================================

class MulticlassDiceLoss(nn.Module):
    """
    Multiclass Dice loss for steel defect segmentation.

    Classes:
        0 = background
        1 = defect class 1
        2 = defect class 2
        3 = defect class 3
        4 = defect class 4

    Background is excluded from the Dice calculation by default.

    Experiment 3:
        Moderate class-aware weighting is used.

        Class 0: 1.00
        Class 1: 2.00
        Class 2: 2.00
        Class 3: 1.25
        Class 4: 1.25
    """

    def __init__(
        self,
        num_classes=5,
        class_weights=None,
        smooth=1.0,
        exclude_background=True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth
        self.exclude_background = exclude_background

        # ----------------------------------------------------
        # Experiment 3 class weights
        # ----------------------------------------------------

        if class_weights is None:
            class_weights = [
                1.0,   # background
                2.0,   # class 1
                2.0,   # class 2
                1.25,  # class 3
                1.25   # class 4
            ]

        self.register_buffer(
            "class_weights",
            torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

    def forward(self, logits, target):
        """
        Args:
            logits:
                [B, C, H, W]

            target:
                [B, H, W]

        Returns:
            Weighted multiclass Dice loss.
        """

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # Convert target to one-hot representation
        # ----------------------------------------------------

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=self.num_classes
        )

        target_one_hot = target_one_hot.permute(
            0,
            3,
            1,
            2
        ).float()

        # ----------------------------------------------------
        # Determine classes used for Dice
        # ----------------------------------------------------

        if self.exclude_background:
            start_class = 1
        else:
            start_class = 0

        dice_losses = []

        # ----------------------------------------------------
        # Calculate Dice for each class
        # ----------------------------------------------------

        for class_id in range(
            start_class,
            self.num_classes
        ):

            pred_class = probabilities[
                :,
                class_id
            ]

            target_class = target_one_hot[
                :,
                class_id
            ]

            # ------------------------------------------------
            # Flatten spatial dimensions
            # ------------------------------------------------

            pred_flat = pred_class.reshape(
                pred_class.shape[0],
                -1
            )

            target_flat = target_class.reshape(
                target_class.shape[0],
                -1
            )

            # ------------------------------------------------
            # Intersection
            # ------------------------------------------------

            intersection = (
                pred_flat * target_flat
            ).sum(dim=1)

            # ------------------------------------------------
            # Dice denominator
            # ------------------------------------------------

            denominator = (
                pred_flat.sum(dim=1)
                +
                target_flat.sum(dim=1)
            )

            # ------------------------------------------------
            # Dice score
            # ------------------------------------------------

            dice = (
                2.0 * intersection
                +
                self.smooth
            ) / (
                denominator
                +
                self.smooth
            )

            # ------------------------------------------------
            # Only calculate Dice for images where the
            # class actually exists in the ground truth.
            # ------------------------------------------------

            valid = (
                target_flat.sum(dim=1) > 0
            )

            if valid.any():

                class_dice = dice[
                    valid
                ].mean()

                class_loss = (
                    1.0 - class_dice
                )

                class_weight = (
                    self.class_weights[class_id]
                )

                dice_losses.append(
                    class_loss * class_weight
                )

        # ----------------------------------------------------
        # Safety case:
        # no foreground classes present
        # ----------------------------------------------------

        if len(dice_losses) == 0:
            return logits.sum() * 0.0

        # ----------------------------------------------------
        # Normalize weighted Dice loss
        # ----------------------------------------------------

        return torch.stack(
            dice_losses
        ).sum() / self.class_weights[
            start_class:
        ].sum()


# ============================================================
# STEEL DEFECT LOSS
# ============================================================

class SteelDefectLoss(nn.Module):
    """
    Experiment 3 loss for steel defect segmentation.

    Total Loss:

        0.60 * Cross Entropy
        +
        0.40 * Dice Loss

    Class-aware weighting:

        Class 0 = 1.00
        Class 1 = 2.00
        Class 2 = 2.00
        Class 3 = 1.25
        Class 4 = 1.25

    The weighting is intentionally moderate so that Classes
    1 and 2 receive additional learning emphasis without
    excessively suppressing Classes 3 and 4.
    """

    def __init__(
        self,
        num_classes=5,
        ce_weight=0.60,
        dice_weight=0.40
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # ----------------------------------------------------
        # Experiment 3 class weights
        # ----------------------------------------------------

        class_weights = [
            1.0,   # background
            2.0,   # class 1
            2.0,   # class 2
            1.25,  # class 3
            1.25   # class 4
        ]

        self.register_buffer(
            "class_weights",
            torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

        # ----------------------------------------------------
        # Weighted Cross Entropy
        # ----------------------------------------------------

        self.cross_entropy = nn.CrossEntropyLoss(
            weight=self.class_weights
        )

        # ----------------------------------------------------
        # Weighted foreground Dice
        # ----------------------------------------------------

        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            smooth=1.0,
            exclude_background=True
        )

    def forward(self, logits, target):
        """
        Args:
            logits:
                [B, C, H, W]

            target:
                [B, H, W]

            or:

            target:
                [B, 1, H, W]

        Returns:
            Combined segmentation loss.
        """

        # ----------------------------------------------------
        # Remove singleton mask channel if present
        # ----------------------------------------------------

        if target.ndim == 4:
            target = target.squeeze(1)

        target = target.long()

        # ----------------------------------------------------
        # Cross Entropy
        # ----------------------------------------------------

        ce_loss = self.cross_entropy(
            logits,
            target
        )

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice_loss = self.dice(
            logits,
            target
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            self.ce_weight * ce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss


# ============================================================
# LOSS COMPONENT CALCULATION
# ============================================================

def calculate_loss_components(
    logits,
    target,
    loss_function
):
    """
    Calculate:

        1. Total loss
        2. Cross Entropy loss
        3. Dice loss

    separately for training logs.

    This function is kept compatible with the existing
    training pipeline.
    """

    # --------------------------------------------------------
    # Remove singleton mask channel if present
    # --------------------------------------------------------

    if target.ndim == 4:
        target = target.squeeze(1)

    target = target.long()

    # --------------------------------------------------------
    # Cross Entropy
    # --------------------------------------------------------

    ce_loss = loss_function.cross_entropy(
        logits,
        target
    )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_loss = loss_function.dice(
        logits,
        target
    )

    # --------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------

    total_loss = (
        loss_function.ce_weight * ce_loss
        +
        loss_function.dice_weight * dice_loss
    )

    return (
        total_loss,
        ce_loss,
        dice_loss
    )


# ============================================================
# LOSS TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Steel Defect Loss - Experiment 3")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dummy data
    # --------------------------------------------------------

    batch_size = 2
    num_classes = 5
    height = 256
    width = 1024

    logits = torch.randn(
        batch_size,
        num_classes,
        height,
        width,
        device=device,
        requires_grad=True
    )

    target = torch.randint(
        0,
        num_classes,
        (
            batch_size,
            height,
            width
        ),
        device=device
    )

    # --------------------------------------------------------
    # Create loss
    # --------------------------------------------------------

    criterion = SteelDefectLoss(
        num_classes=5
    ).to(device)

    # --------------------------------------------------------
    # Display class weights
    # --------------------------------------------------------

    print()
    print("Class weights:")
    print(
        criterion.class_weights.detach().cpu().tolist()
    )

    # --------------------------------------------------------
    # Calculate loss components
    # --------------------------------------------------------

    total_loss, ce_loss, dice_loss = (
        calculate_loss_components(
            logits,
            target,
            criterion
        )
    )

    print()
    print(
        f"Total loss: {total_loss.item():.6f}"
    )

    print(
        f"CE loss:    {ce_loss.item():.6f}"
    )

    print(
        f"Dice loss:  {dice_loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    total_loss.backward()

    print()
    print(
        "Backward pass successful"
    )

    print()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert torch.isfinite(
        total_loss
    ).item()

    assert logits.grad is not None

    print(
        "Loss test PASSED"
    )

    print("=" * 60)