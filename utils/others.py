"""Small helpers used by the FluxSR training objective."""

import torch
from torch import nn
import torchvision.transforms as transforms


def cycle(dl):
    while True:
        for data in dl:
            yield data


class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super(EdgeDetectionModel, self).__init__()
        # Sobel filters for edge detection
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])
        sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]])

        self.sobel_x.weight = nn.Parameter(sobel_x_kernel.view(1, 1, 3, 3))
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel.view(1, 1, 3, 3))
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            x = transforms.Grayscale()(x)

        # Apply Sobel filters
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)

        # Calculate gradient magnitude (edge detection result)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edges


def total_variation_loss(x):
    """Return the (B, C, H-1, W-1) anisotropic TV map used by TV-LPIPS."""
    diff_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_y = x[:, :, :, 1:] - x[:, :, :, :-1]
    loss = torch.abs(diff_x)[..., :-1] + torch.abs(diff_y)[..., :-1, :]
    return loss


def attention_diversification_loss(feature, eps=1e-8):
    """Mean cosine similarity between image tokens and their mean token.

    Accepts (batch, tokens, channels) or (batch, layers, tokens, channels).
    Low-precision features are reduced in float32; epsilon defines a finite
    cosine for zero-norm tokens.
    """
    if feature.dtype in (torch.float16, torch.bfloat16):
        feature = feature.float()
    mean_feature = feature.mean(dim=-2, keepdim=True)
    return torch.nn.functional.cosine_similarity(
        feature, mean_feature, dim=-1, eps=eps
    ).mean()
