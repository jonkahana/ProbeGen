import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torchvision

DEAD_LEAVES_DATASET_PATH = 'experiments/cnn_generalization/dataset/dead_leaves-textures/train'

class Deep_Linear_Generator(nn.Module):
    def __init__(self, out_c, nz=100, width_mult=16, n_layers=6):
        super(Deep_Linear_Generator, self).__init__()
        self.main = nn.ModuleList([])

        if n_layers == 6:
            self.main.append(nn.ConvTranspose2d(nz, width_mult * 8, 4, 1, 0, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult * 8, width_mult * 4, 4, 2, 1, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult * 4, width_mult * 2, 4, 2, 1, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult * 2, width_mult * 2, 3, 1, 1, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult * 2, width_mult, 4, 2, 1, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult, out_c, 3, 1, 1, bias=True))
            self.main.append(nn.Tanh())
            self.main = nn.Sequential(*self.main)
        if n_layers == 5:
            self.main.append(nn.ConvTranspose2d(nz, width_mult * 8, 4, 1, 0, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult * 8, width_mult * 4, 4, 2, 1, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult * 4, width_mult * 2, 4, 2, 1, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult * 2, width_mult, 4, 2, 1, bias=True))
            self.main.append(nn.ConvTranspose2d(width_mult, out_c, 3, 1, 1, bias=True))
            self.main.append(nn.Tanh())
            self.main = nn.Sequential(*self.main)
        if n_layers == 0:
            self.main = nn.Tanh()

    def forward(self, input):
        return self.main(input)


class ProbeGen(nn.Module):

    def __init__(self, n_tokens, d_hidden, models_c_in, models_c_out, d_out,
                 gen_type='deep_linear_5', gen_latent_z=32, gen_width=16, n_layers=6):
        super().__init__()
        self.n_tokens = n_tokens
        self.models_c_in = models_c_in
        self.models_c_out = models_c_out

        self.gen_type = gen_type
        self.gen_latent_z = gen_latent_z
        self.gen_hidden = gen_width

        if 'no_opt' not in self.gen_type:
            self.input = nn.Parameter(torch.randn(n_tokens, self.gen_latent_z, 1, 1))

        if self.gen_type == 'deep_linear_6':
            self.tokens_generator = Deep_Linear_Generator(self.models_c_in, nz=self.gen_latent_z, width_mult=self.gen_hidden, n_layers=6)
        elif self.gen_type == 'deep_linear_5':
            self.tokens_generator = Deep_Linear_Generator(self.models_c_in, nz=self.gen_latent_z, width_mult=self.gen_hidden, n_layers=5)
        elif self.gen_type == 'deep_linear_0':
            self.input = nn.Parameter(torch.randn(n_tokens, self.models_c_in, 32, 32))
            self.tokens_generator = Deep_Linear_Generator(self.models_c_in, nz=self.gen_latent_z, width_mult=self.gen_hidden, n_layers=0)

        elif self.gen_type == 'dead_leaves__no_opt':
            img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                                                                             std=[0.24703223, 0.24348513, 0.26158784])
                                                            ])
            if self.models_c_in == 1:
                img_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1), torchvision.transforms.ToTensor()])

            dataset = torchvision.datasets.ImageFolder(DEAD_LEAVES_DATASET_PATH, transform=img_transform)
            chosen_inds = train_test_split(np.arange(len(dataset)), test_size=n_tokens)[1]
            token_imgs = [dataset[i][0].unsqueeze(0) for i in chosen_inds]
            self.input = torch.cat(token_imgs, dim=0).to('cuda')
            self.input.requires_grad = False
            self.tokens_generator = nn.Identity()


        elif self.gen_type == 'linear_0_no_acts':
            self.input = nn.Parameter(torch.randn(n_tokens, self.models_c_in))
            self.tokens_generator = nn.Identity()
        elif self.gen_type == 'linear_2_no_acts':
            self.input = nn.Parameter(torch.randn(n_tokens, self.gen_latent_z))
            self.tokens_generator = nn.Sequential(nn.Linear(self.gen_latent_z, self.gen_latent_z), nn.Linear(self.gen_latent_z, self.models_c_in))

        elif self.gen_type == 'uniform_coords__no_opt':
            self.input = torch.rand((n_tokens, self.models_c_in))
            self.input = self.input * 2 - 1
            self.input = self.input.to('cuda')
            self.tokens_generator = nn.Identity()
            self.input.requires_grad = False

        else:
            raise ValueError(f"Generator type: {self.gen_type} not recognized")

        p_emb_in = self.models_c_out * self.n_tokens
        self.points_mixer = nn.ModuleList([nn.Linear(p_emb_in, d_hidden), nn.ReLU()])
        for i in range(n_layers - 2):
            self.points_mixer.append(nn.Linear(d_hidden, d_hidden))
            self.points_mixer.append(nn.ReLU())
        self.points_mixer.append(nn.Linear(d_hidden, d_out))
        self.points_mixer = nn.Sequential(*self.points_mixer)

    def forward_generator(self, nets):
        x = self.tokens_generator(self.input)
        x = torch.cat([net(x).unsqueeze(0) for net in nets], dim=0)
        return x

    def forward(self, nets):
        x = self.tokens_generator(self.input)
        x = torch.cat([net(x).unsqueeze(0) for net in nets], dim=0)
        x = x.view(x.shape[0], -1)
        x = self.points_mixer(x)
        return x


class Probing_Gen_Adapter(nn.Module):
    def __init__(self, n_tokens, d_hidden, models_c_out, d_out, n_layers=6):
        super().__init__()

        self.n_tokens = n_tokens
        self.models_c_out = models_c_out

        p_emb_in = self.models_c_out * self.n_tokens
        self.points_mixer = nn.ModuleList([nn.Linear(p_emb_in, d_hidden), nn.ReLU()])
        for i in range(n_layers - 2):
            self.points_mixer.append(nn.Linear(d_hidden, d_hidden))
            self.points_mixer.append(nn.ReLU())
        self.points_mixer.append(nn.Linear(d_hidden, d_out))
        self.points_mixer = nn.Sequential(*self.points_mixer)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.points_mixer(x)
        return x
