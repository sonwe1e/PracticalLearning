import torch
import torch.nn as nn
import torch.nn.functional as F


class vector_quantizer(nn.Module):
    def __init__(self, n_e, n_dim, beta):
        super(vector_quantizer, self).__init__()
        self.n_e = n_e
        self.n_dim = n_dim
        self.beta = beta
        # nn.Embedding 类似于 nn.Parameter
        self.embedding = nn.Embedding(self.n_e, self.n_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_e, 1 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.n_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings e_j
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = (
            torch.zeros(min_encoding_indices.shape[0], self.n_e)
            .to(z.device)
            .scatter_(1, min_encoding_indices, 1)
        )

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # calculate the times that each embedding vector is used
        e_mean = torch.mean(min_encodings, dim=0)
        # perplexity is a metric to measure how well a probability distribution
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class vqvae(nn.Module):
    def __init__(self, conf):
        super(vqvae, self).__init__()
        self.channels = conf["channels"]
        self.channels_div = conf["channels_div"]
        self.channels = [c // self.channels_div for c in self.channels]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.channels[i],
                        self.channels[i + 1],
                        kernel_size=7,
                        stride=2,
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
        self.vector_quantizer = vector_quantizer(
            conf["n_embeddings"], self.channels[-1], conf["beta"]
        )
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
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

    def forward(self, x):
        x = self.start_layer(x)
        z = self.encoder(x)
        (
            loss,
            z_q,
            perplexity,
            min_encodings,
            min_encoding_indices,
        ) = self.vector_quantizer(z)
        r_image = self.decoder(z_q)
        r_image = 3 * self.finnal_layer(r_image)
        return r_image, z, loss

    def loss_function(self, x, r_image):
        recons_loss = F.mse_loss(r_image, x)
        return recons_loss

    def sample(self, n):
        # z = torch.rand(n, self.channels[-1], 4, 4)
        # _, z, _, _, _ = self.vector_quantizer(z)
        rand_indices = torch.randint(0, self.vector_quantizer.n_e, (n * 16,))
        z = self.vector_quantizer.embedding(rand_indices)
        z = z.view(n, self.channels[-1], 4, 4)
        r_image = self.decoder(z)
        r_image = 3 * self.finnal_layer(r_image)
        return r_image


if __name__ == "__main__":
    import yaml

    with open("./config/vqvae.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    model = vqvae(conf)
    print(model)
    x = torch.randn(2, 3, conf["image_size"], conf["image_size"])
    print(x.shape)
    r_image, z, loss = model(x)
    print(r_image.shape, z.shape)
    samples = model.sample(10)
    print(samples.shape)
