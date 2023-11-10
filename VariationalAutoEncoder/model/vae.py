import torch
import torch.nn as nn
import torch.nn.functional as F


class vae(nn.Module):
    def __init__(self, conf):
        super(vae, self).__init__()
        self.channels = conf["channels"]
        self.channels_div = conf["channels_div"]
        self.channels = [c // self.channels_div for c in self.channels]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.latent_dim = conf["latent_dim"]
        for i in range(len(self.channels) - 1):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.channels[i],
                        self.channels[i + 1],
                        kernel_size=7,
                        stride=4,
                        padding=3,
                    ),
                    nn.BatchNorm2d(self.channels[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        for i in range(len(self.channels) - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.channels[i],
                        self.channels[i - 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(self.channels[i - 1]),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        self.mean_layer = nn.Linear(self.channels[-1], self.latent_dim)
        self.logvar_layer = nn.Linear(self.channels[-1], self.latent_dim)
        self.fc = nn.Linear(self.latent_dim, self.channels[-1] * 2 * 2)
        self.start_layer = nn.Sequential(
            nn.Conv2d(3, self.channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channels[0]),
            nn.LeakyReLU(),
        )
        self.finnal_layer = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels[0],
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(self.channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(self.channels[0], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        x = self.start_layer(x)
        hidden = self.encoder(x)
        hidden = torch.flatten(hidden, start_dim=1)
        mu = self.mean_layer(hidden)
        logvar = self.logvar_layer(hidden)
        z = self.reparameterize(mu, logvar)
        z = self.fc(z)
        r_image = z.view(-1, self.channels[-1], 2, 2)
        r_image = self.decoder(r_image)
        r_image = 3 * self.finnal_layer(r_image)
        return r_image, z, mu, logvar

    def loss_function(self, x, r_image, mu, log_var):
        recons_loss = F.mse_loss(r_image, x)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        loss = recons_loss + 0.0025 * kld_loss
        return loss, recons_loss, kld_loss

    def sample(self, n):
        z = torch.randn(n, self.latent_dim)
        z = self.fc(z)
        r_image = z.view(-1, self.channels[-1], 2, 2)
        r_image = self.decoder(r_image)
        r_image = self.finnal_layer(r_image)
        return r_image


if __name__ == "__main__":
    import yaml

    with open("../config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    model = vae(conf)
    print(model)
    x = torch.randn(2, 3, conf["image_size"], conf["image_size"])
    r_img, z, mu, logvar = model(x)
    print(r_img.shape, z.shape, mu.shape, logvar.shape)
