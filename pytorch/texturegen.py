"""
Implement the texture synthesis pipeline(same resolution, without spatial tag) of the paper in PyTorch
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from time import time

# import pytorch-related packages
import torch
import torchvision
from PIL import Image
from vgg_model import VGG19Model
from loss_fn import Slicing_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = VGG19Model(device, 3)  # customized_vgg_model in pytorch

parser = argparse.ArgumentParser(description='Texture generation')
# parser.add_argument('filename', help='filename of the input texture')
parser.add_argument('--size', default=256, type=int, help='resolution of the input texture (it will resize to this resolution)')
parser.add_argument('--output', default='output.jpg', help='name of the output file')
parser.add_argument('--iters', default=20, type=int, help='number of steps')
parser.add_argument('--output_folder', default='../outputs', type=str, help='output folder')
parser.add_argument('--data_folder', default='SlicedW', type=str, help='data folder')
parser.add_argument('--img_name', default='input.jpg', type=str, help='data folder')
parser.add_argument('--log_freq', default=2, type=int, help='logging frequency')
args = parser.parse_args()

SIZE = args.size
# INPUT_FILE = args.filename
OUTPUT_FILE = args.output
INPUT_FILE = '../data/' + args.data_folder + '/' + args.img_name
NB_ITER = args.iters

output_folder = args.output_folder + '/' + args.img_name
results_folder = '../results/' + args.img_name

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

print(f'Launching texture synthesis from {INPUT_FILE} on size {SIZE} for {NB_ITER} steps. Output file: {OUTPUT_FILE}')


def decode_image(path, size=SIZE):
    """ Load and resize the input texture """

    img = Image.open(path).convert("RGB")
    width, height = img.size
    max_size = width
    if max_size > height:
        max_size = height
    trans = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(max_size),
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
    ])
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    # print('torch:', img.shape, img.min(), img.max(), path)
    return img


def fit(nb_iter, texture, extractor):
    targets = extractor(texture)
    slicing_torch = Slicing_torch(device, targets, repeat_rate=1)  # target_sorted_sliced

    # Image initialization  
    image = np.zeros((1, 3, SIZE, SIZE))
    image = image.astype(np.float32)
    image = torch.from_numpy(image).to(device)
    image = image + torch.mean(texture, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
    image = image + (torch.randn(size=(1, 3, SIZE, SIZE)).to(device) * 1e-2)
    plt.imsave(output_folder + '/init.jpg', image.permute(0, 2, 3, 1).detach().cpu().numpy()[0])

    optimizer = torch.optim.LBFGS([image.requires_grad_()], lr=1, max_iter=50, max_eval=64, tolerance_grad=0, tolerance_change=0)
    # optimizer = torch.optim.Adam([image.requires_grad_()], lr=0.02)   # use adam

    for i in range(nb_iter):
        def closure():
            optimizer.zero_grad()
            image.data.clamp_(0, 1)
            # image.data.clamp_(0, 1)
            output_features = extractor(image)
            loss = slicing_torch(output_features)
            # print(f'iter {i + 1} loss {loss}')
            loss.backward()
            return loss

        optimizer.step(closure)

        if i % args.log_freq == 0 or i == nb_iter - 1:
            image.data.clamp_(0, 1)
            output_features = extractor(image)
            loss = slicing_torch(output_features)

            print(f'iter {i + 1} loss {loss}')
            plt.imsave(output_folder + '/output-iter{:}.jpg'.format(i + 1), image.permute(0, 2, 3, 1)[0].detach().cpu().numpy())

            for j in range(len(output_features)):

                # just to visualize feature maps of a specific layer
                if j == 7:
                    # print('targets:', targets[j][0].shape, output_features[j][0].shape)
                    target_grid = torchvision.utils.make_grid(targets[j][0].unsqueeze(1), nrow=16, padding=10)
                    output_grid = torchvision.utils.make_grid(output_features[j][0].unsqueeze(1), nrow=16, padding=10)
                    target_grid_save = target_grid.detach().cpu().numpy().transpose(1, 2, 0)
                    target_grid_save = np.clip(target_grid_save / target_grid_save.max(), 0, 1)
                    plt.imsave(output_folder + '/target_grid.jpg', target_grid_save)
                    plt.clf()

                    output_grid_save = output_grid.detach().cpu().numpy().transpose(1, 2, 0)
                    output_grid_save = np.clip(output_grid_save / output_grid_save.max(), 0, 1)
                    plt.imsave(output_folder + '/output_grid.jpg', output_grid_save)
                    plt.clf()

        # Change random directions (optional)
        slicing_torch.update_slices(targets)

    return image


texture_reference = decode_image(INPUT_FILE, size=SIZE)
# export of the resized input: reference to compare against the generated texture
plt.imsave(output_folder + '/resized-input.jpg', texture_reference.permute(0, 2, 3, 1).detach().cpu().numpy()[0])
plt.imsave(results_folder + '/resized-input.jpg', texture_reference.permute(0, 2, 3, 1).detach().cpu().numpy()[0])

print('start fitting')
start = time()
output_image = fit(NB_ITER, texture_reference, extractor)
plt.imsave(output_folder + '/'+OUTPUT_FILE, output_image[0].permute(1, 2, 0).detach().cpu().numpy())
plt.imsave(results_folder + '/'+OUTPUT_FILE, output_image[0].permute(1, 2, 0).detach().cpu().numpy())
end = time()
print('Time: {:} minutes'.format((end - start) / 60.0))
