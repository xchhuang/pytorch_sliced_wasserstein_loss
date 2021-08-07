"""
Implement the slicing operation in pytorch, also make comparison with tf2 version
"""
import torch
import numpy as np
import torch.nn.functional as F


class Slicing_torch(torch.nn.Module):
    def __init__(self, device, layers, repeat_rate):
        super().__init__()
        # Number of directions
        self.device = device
        self.repeat_rate = repeat_rate
        self.update_slices(layers)

    def update_slices(self, layers):
        directions = []
        for l in layers:    # converted to [B, W, H, D]
            if l.ndim == 4:
                l = l.permute(0, 2, 3, 1)
            if l.ndim == 5:
                l = l.permute(0, 2, 3, 4, 1)

            dim_slices = l.shape[-1]
            num_slices = l.shape[-1]
            # num_slices = 512
            # print('num_slices:', num_slices, dim_slices)
            cur_dir = torch.randn(size=(num_slices, dim_slices)).to(self.device)
            norm = torch.sqrt(torch.sum(torch.square(cur_dir), axis=-1))
            norm = norm.view(num_slices, 1)
            cur_dir = cur_dir / norm
            directions.append(cur_dir)
        self.directions = directions
        self.target = self.compute_target(layers)


    def compute_proj(self, input, layer_idx, repeat_rate):
        if input.ndim == 4:
            input = input.permute(0, 2, 3, 1)
        if input.ndim == 5:
            input = input.permute(0, 2, 3, 4, 1)

        batch = input.size(0)
        dim = input.size(-1)
        tensor = input.view(batch, -1, dim)
        tensor_permute = tensor.permute(0, 2, 1)

        # Project each pixel feature onto directions (batch dot product)
        sliced = torch.matmul(self.directions[layer_idx], tensor_permute)
        # print('sliced(torch):', sliced.shape, self.repeat_rate)

        # # Sort projections for each direction
        sliced, _ = torch.sort(sliced)
        sliced = sliced.repeat_interleave(repeat_rate ** 2, dim=-1)
        sliced = sliced.view(batch, -1)
        return sliced

    def compute_target(self, layers):
        target = []
        # target_sorted_sliced = []
        for idx, l in enumerate(layers):
            # target_sorted_sliced.append(l)
            sliced_l = self.compute_proj(l, idx, self.repeat_rate)
            target.append(sliced_l.detach())
        return target

    def forward(self, input):
        loss = 0.0
        # output = []
        for idx, l in enumerate(input):
            cur_l = self.compute_proj(l, idx, 1)
            # output.append(l)
            loss += F.mse_loss(cur_l, self.target[idx])
        return loss
