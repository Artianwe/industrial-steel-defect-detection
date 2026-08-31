import torch

from model import create_model


print("=" * 60)
print("GPU MODEL TEST")
print("=" * 60)


# ------------------------------------------------------------
# Check CUDA
# ------------------------------------------------------------

assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda")

print(f"GPU: {torch.cuda.get_device_name(0)}")


# ------------------------------------------------------------
# Clear GPU memory
# ------------------------------------------------------------

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------
# Create model
# ------------------------------------------------------------

model = create_model(
    num_classes=5,
    pretrained=False
)

model = model.to(device)

model.train()


# ------------------------------------------------------------
# Batch size
# ------------------------------------------------------------

batch_size = 2

print(f"Batch size: {batch_size}")


# ------------------------------------------------------------
# Create dummy input
# ------------------------------------------------------------

images = torch.randn(
    batch_size,
    3,
    256,
    1024,
    device=device
)

masks = torch.randint(
    0,
    5,
    (
        batch_size,
        256,
        1024
    ),
    device=device
)


# ------------------------------------------------------------
# Import loss
# ------------------------------------------------------------

from losses import create_loss

criterion = create_loss()

criterion = criterion.to(device)


# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

print("\nRunning forward pass...")

outputs = model(images)

print(
    f"Output shape: {outputs.shape}"
)


# ------------------------------------------------------------
# Calculate loss
# ------------------------------------------------------------

print("Calculating loss...")

loss, ce_loss, dice_loss = criterion(
    outputs,
    masks
)

print(
    f"Loss: {loss.item():.6f}"
)


# ------------------------------------------------------------
# Backward pass
# ------------------------------------------------------------

print("Running backward pass...")

loss.backward()


# ------------------------------------------------------------
# GPU memory
# ------------------------------------------------------------

torch.cuda.synchronize()

allocated = (
    torch.cuda.memory_allocated()
    / 1024**3
)

reserved = (
    torch.cuda.memory_reserved()
    / 1024**3
)

peak = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)


print("\nGPU MEMORY")
print("-" * 40)

print(
    f"Currently allocated: {allocated:.2f} GB"
)

print(
    f"Currently reserved:  {reserved:.2f} GB"
)

print(
    f"Peak allocated:      {peak:.2f} GB"
)


# ------------------------------------------------------------
# Clean up
# ------------------------------------------------------------

del outputs
del loss
del ce_loss
del dice_loss
del images
del masks
del model

torch.cuda.empty_cache()


print("\nGPU TEST PASSED")
print("=" * 60)