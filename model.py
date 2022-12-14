import torch
import torch.nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision
import torch
from torch import nn, relu
from torchvision import datasets, transforms
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from time import time

LOG_2PI = np.log(2 * np.pi)


class pPCA(object):
    def __init__(self, data, latent_dim):
        self.data = data
        self.k = latent_dim
        self.n = data.shape[1]
        self.N = data.shape[0]
        self.covmat = np.cov(self.data, rowvar=False)
        self.eigenvalue, self.eigenvector = np.linalg.eigh(self.covmat)
        self.eigenvalue = self.eigenvalue[::-1]
        self.eigenvector = self.eigenvector[:, ::-1]
        self.R = np.eye(self.k)
        self.W_mle = None
        self.sigma_mle = None
        self.loglikelihood = None

    def W_MLE(self):
        sigma_mle = self.sigma_mle
        Lambda_k = np.diag(self.eigenvalue[:self.k])
        U_k = self.eigenvector[:, :self.k]
        self.W_mle = U_k @ (Lambda_k - sigma_mle * np.eye(self.k)) ** (0.5) @ self.R
        return self.W_mle

    def sigma_MLE(self):
        sum_lost_eigenvalue = np.sum(self.eigenvalue[self.k:])
        self.sigma_mle = sum_lost_eigenvalue / (self.n - self.k)
        return self.sigma_mle

    def log_likelihood(self):
        C = self.W_mle @ (self.W_mle.T) + self.sigma_mle * np.eye(self.W_mle.shape[0])
        self.loglikelihood = -self.N / 2 * (
                self.n * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + np.trace(np.linalg.inv(C) @ self.covmat)) / (
                                 self.N)
        return self.loglikelihood

    def calculate_log_likelihood_with_fixed_sigma(self, sigma):
        sigma_mle = sigma
        Lambda_k = np.diag(self.eigenvalue[:self.k])
        U_k = self.eigenvector[:, :self.k]
        temp = Lambda_k - sigma_mle * np.eye(self.k)
        temp[temp < 0] = 0
        W_mle = U_k @ (temp) ** (0.5) @ self.R
        C = W_mle @ (W_mle.T) + sigma_mle * np.eye(W_mle.shape[0])
        loglikelihood = -self.N / 2 * (
                self.n * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + np.trace(np.linalg.inv(C) @ self.covmat)) / (
                            self.N)
        return loglikelihood

    def get_result(self):
        sigma_mle = self.sigma_MLE()
        W_mle = self.W_MLE()
        loglikelihood = self.log_likelihood()
        return W_mle, sigma_mle, loglikelihood


class LinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, sigma_init, sigma_trainable, mode='Analytic'):
        super(LinearVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.outputDim = output_dim
        self.mode = mode
        # Learnable parameters
        self.logD = nn.Parameter(torch.randn(self.latent_dim))
        self.V = nn.Parameter(torch.randn(self.input_dim, self.latent_dim) / np.sqrt(self.input_dim))
        self.mu = nn.Parameter(torch.randn(self.input_dim))
        self.W = nn.Parameter(
            torch.randn(self.latent_dim, self.outputDim) / np.sqrt(self.latent_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(sigma_init), requires_grad=sigma_trainable)

    def kl_div(self, q_mean, q_logvar, p_mean, p_logvar):
        return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2) / torch.exp(p_logvar) - 1)

    def Analytic(self, x):
        z_mean = (x - self.mu) @ (self.V)
        x = x - self.mu
        temp1 = torch.trace(self.W.T @ (torch.exp(self.logD).unsqueeze(1) * self.W))
        temp2 = torch.sum((x @ (self.V) @ (self.W)) ** 2, dim=1)
        temp3 = torch.sum((x @ (self.V) @ (self.W)) * x, dim=1)
        temp4 = torch.sum(x ** 2, dim=1)
        temp5 = self.input_dim * (np.log(2 * np.pi) + self.log_sigma_sq) / 2
        rec_loss = (((temp1 + temp2 - 2.0 * temp3 + temp4) / torch.exp(self.log_sigma_sq)) / 2 + temp5).mean()
        kl_loss = self.kl_div(z_mean, self.logD.unsqueeze(0), torch.tensor(0.0), torch.tensor(0.0)).sum() / len(x)
        elbo = rec_loss + kl_loss
        return elbo

    def Stochastic(self, x):
        z_mean = (x - self.mu) @ (self.V)
        try:
            noise = torch.randn(len(x), self.latent_dim).to('cuda:0')
        except:
            noise = torch.randn(len(x), self.latent_dim)
        z_sample = z_mean + noise * torch.exp(self.logD / 2)
        x_mean = z_sample @ self.W + self.mu
        rec_loss = (0.5 * (torch.sum((x - x_mean) ** 2 / torch.exp(self.log_sigma_sq), 1) + (
                np.log(2 * np.pi) + self.log_sigma_sq) * self.input_dim)).mean()
        kl_loss = self.kl_div(z_mean, self.logD.unsqueeze(0), torch.tensor(0.0), torch.tensor(0.0)).sum() / len(x)
        elbo = rec_loss + kl_loss
        return elbo

    def forward(self, x):
        if self.mode == 'Analytic':
            elbo = self.Analytic(x)
            return elbo
        elif self.mode == 'Stochastic':
            elbo = self.Stochastic(x)
            return elbo
        # q_distribution = torch.distributions.Normal(q_mean, std_batch)
        # z = q_distribution.rsample()
        #
        # p_mean = torch.matmul(z.unsqueeze(1), W_batch).squeeze(1) + mu_batch
        # p_distribution = torch.distributions.Normal(p_mean, self.sigma)
        #
        # rex = p_distribution.rsample()


class NonLinearEncoderLinearDecoderVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, sigma_init, sigma_trainable):
        super(NonLinearEncoderLinearDecoderVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 28 * 28),
        )
        self.log_sigma_sq = nn.Parameter(torch.tensor(sigma_init), requires_grad=sigma_trainable)

    def kl_div(self, q_mean, q_logvar, p_mean, p_logvar):
        return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2) / torch.exp(p_logvar) - 1)

    def forward(self, x):
        temp = self.encoder(x)

        # variational
        mu = self.fc_mu(temp)
        log_var = self.fc_logvar(temp)

        std = torch.exp(log_var / 2)
        try:
            noise = torch.randn(len(x), self.latent_dim).to('cuda:0')
        except:
            noise = torch.randn(len(x), self.latent_dim)
        z_sample = mu + noise * std

        rec_x = self.decoder(z_sample)
        rec_loss = (0.5 * (torch.sum((rec_x - x) ** 2 / torch.exp(self.log_sigma_sq), 1) + (
                np.log(2 * np.pi) + self.log_sigma_sq) * self.input_dim)).mean()
        kl_loss = self.kl_div(mu, log_var, torch.tensor(0.0), torch.tensor(0.0)).sum() / len(x)
        elbo = rec_loss + kl_loss

        return elbo, self.log_sigma_sq


class DeepLinearEncoderLinearDecoderVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, sigma_init, sigma_trainable):
        super(DeepLinearEncoderLinearDecoderVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable

        self.encoder_fc_mu = nn.Linear(28 * 28, self.latent_dim)
        self.encoder_fc_logvar = nn.Linear(28 * 28, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 28 * 28),
        )
        self.log_sigma_sq = nn.Parameter(torch.tensor(sigma_init), requires_grad=sigma_trainable)

    def kl_div(self, q_mean, q_logvar, p_mean, p_logvar):
        return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2) / torch.exp(p_logvar) - 1)

    def forward(self, x):

        # variational
        mu = self.encoder_fc_mu(x)
        log_var = self.encoder_fc_logvar(x)
        std = torch.exp(log_var / 2)
        try:
            noise = torch.randn(len(x), self.latent_dim).to('cuda:0')
        except:
            noise = torch.randn(len(x), self.latent_dim)
        z_sample = mu + noise * std

        rec_x = self.decoder(z_sample)
        rec_loss = (0.5 * (torch.sum((rec_x - x) ** 2 / torch.exp(self.log_sigma_sq), 1) + (
                np.log(2 * np.pi) + self.log_sigma_sq) * self.input_dim)).mean()
        kl_loss = self.kl_div(mu, log_var, torch.tensor(0.0), torch.tensor(0.0)).sum() / len(x)
        elbo = rec_loss + kl_loss

        return elbo, self.log_sigma_sq


class DeepNonLinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, sigma_init, sigma_trainable):
        super(DeepNonLinearVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
        )
        self.log_sigma_sq = nn.Parameter(torch.tensor(sigma_init), requires_grad=sigma_trainable)

    def kl_div(self, q_mean, q_logvar, p_mean, p_logvar):
        return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2) / torch.exp(p_logvar) - 1)

    def forward(self, x):
        temp = self.encoder(x)

        # variational
        mu = self.fc_mu(temp)
        log_var = self.fc_logvar(temp)
        std = torch.exp(log_var / 2)
        try:
            noise = torch.randn(len(x), self.latent_dim).to('cuda:0')
        except:
            noise = torch.randn(len(x), self.latent_dim)
        z_sample = mu + noise * std

        rec_x = self.decoder(z_sample)
        rec_loss = (0.5 * (torch.sum((rec_x - x) ** 2 / torch.exp(self.log_sigma_sq), 1) + (
                np.log(2 * np.pi) + self.log_sigma_sq) * self.input_dim)).mean()
        kl_loss = self.kl_div(mu, log_var, torch.tensor(0.0), torch.tensor(0.0)).sum() / len(x)
        elbo = rec_loss + kl_loss

        return elbo, self.log_sigma_sq


class DeepNonLinearVAEForPosteriorCollapse(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, sigma_init, sigma_trainable):
        super(DeepNonLinearVAEForPosteriorCollapse, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
        )
        self.log_sigma_sq = nn.Parameter(torch.tensor(sigma_init), requires_grad=sigma_trainable)

    def kl_div(self, q_mean, q_logvar, p_mean, p_logvar):
        return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2) / torch.exp(p_logvar) - 1)

    def forward(self, x):
        temp = self.encoder(x)

        # variational
        mu = self.fc_mu(temp)
        log_var = self.fc_logvar(temp)
        std = torch.exp(log_var / 2)
        try:
            noise = torch.randn(len(x), self.latent_dim).to('cuda:0')
        except:
            noise = torch.randn(len(x), self.latent_dim)
        z_sample = mu + noise * std

        rec_x = self.decoder(z_sample)
        rec_loss = (0.5 * (torch.sum((rec_x - x) ** 2 / torch.exp(self.log_sigma_sq), 1) + (
                np.log(2 * np.pi) + self.log_sigma_sq) * self.input_dim)).mean()
        kl_loss = self.kl_div(mu, log_var, torch.tensor(0.0), torch.tensor(0.0)).sum() / len(x)
        elbo = rec_loss + kl_loss
        return elbo, self.log_sigma_sq, mu, std


if __name__ == '__main__':
    class config:
        latent_dim = 200
        data_size = 1000
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        epsilon_list = np.linspace(0, 2, 100)
        delta = 0.01


    def seed_torch(seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)


    seed_torch(0)
    latent_dim = config.latent_dim
    data_size = config.data_size
    device = config.device
    delta = config.delta
    epsilon_list = config.epsilon_list

    mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    # loader = mnist_data.data[:data_size].view(-1, 784) / 255.0
    # loader = loader.to(device)
    elbo_result = []


    def preprocess(x, eps):
        x = (x + np.random.rand(*x.shape)) / 256.0
        x = eps + (1 - 2 * eps) * x
        x = np.log(x / (1.0 - x))
        x = x.to(torch.float32)
        return x


    loader = mnist_data.data[:data_size].view(-1, 784)
    loader = preprocess(loader, 1e-6)
    loader_numpy = loader.numpy()
    loader = loader.to(device)

    model = DeepNonLinearVAEForPosteriorCollapse(784, latent_dim, 784, 1.0, True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    start = time()


    def calculate_posterior_collapse(mu, std, epsilon_list, delta):
        all_kl = 0.5 * (mu ** 2 + std ** 2 - 1.0 - torch.log(std ** 2))
        threshold = 1.0 - delta
        each_epsilon_result = []
        for epsilon in epsilon_list:
            if_posterior_collapse = (all_kl < epsilon) + 0.0
            posterior_collapse_probability_for_each_dim = torch.mean(if_posterior_collapse, dim=0)
            percentage_posterior_collapse = ((posterior_collapse_probability_for_each_dim > threshold) + 0.0).mean()
            each_epsilon_result.append(percentage_posterior_collapse.detach().cpu().numpy())
        return each_epsilon_result

    result =[]
    for i in range(1000):
        elbo, log_sigma_sq, mu, std = model(loader)
        optimizer.zero_grad()
        elbo.backward()
        optimizer.step()
        if i == 999:
            result = calculate_posterior_collapse(mu, std, epsilon_list, delta)
            plt.plot(epsilon_list, result, label="LinearEncoder", color='red')
            plt.show()
