import torch.nn as nn
import torch



class Autoencodermodel(nn.Module):
    def __init__(self):
        super(Autoencodermodel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(256, 200, kernel_size=3),
            nn.ReLU(),
            GroupNorm(200),
            nn.Conv2d(200, 150, kernel_size=3,stride=2),
            nn.ReLU(),
            GroupNorm(150,num_groups=30),
            nn.Conv2d(150,100,kernel_size=3),
            nn.ReLU(),
            GroupNorm(100,num_groups=10),
            nn.Conv2d(100,70,kernel_size=2),
            nn.ReLU(),
            GroupNorm(70,num_groups=10),
            nn.Conv2d(70,60,kernel_size=2),
            nn.ReLU(),
            GroupNorm(60,num_groups=20),
            nn.Conv2d(60, 50, kernel_size=1),
            nn.Tanh()

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(50, 150, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(150, 200, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(200, 256, kernel_size=3),
            nn.Tanh()

        )

        self.img_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 200, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(200, 180, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(180, 150, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(150, 128, kernel_size=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ## learns the data representation from input
        z = self.encoder(x)
        ## reconstruct the data based on the learned data representation
        y = self.decoder(z)
        # # reconstruct the images based on the learned data representation
        img = self.img_decoder(y)

        return z,y,img


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
        self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.num_groups ,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        # normalize
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.gamma + self.beta

