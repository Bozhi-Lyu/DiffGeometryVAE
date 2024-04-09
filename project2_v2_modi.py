# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen and Søren Hauberg, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
import torch.utils.data
from tqdm import tqdm
torch.manual_seed(0)
import matplotlib
import matplotlib.pyplot as plt
import random
matplotlib.use('TkAgg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.ContinuousBernoulli(logits=logits), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def getencoder(self):
        return self.encoder
    
    def getdecoder(self):
        return self.decoder

    def getprior(self):
        return self.prior

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

def noise(x, std=0.05):
    eps = std * torch.randn_like(x)
    return torch.clamp(x + eps, min=0.0, max=1.0)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader)*epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = noise(x.to(device))
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1


def proximity(curve_points, latent):
    """
    Compute the average distance between points on a curve and a collection
    of latent variables.

    Parameters:
    curve_points: [torch.tensor]
        M points along a curve in latent space. tensor shape: M x latent_dim
    latent: [torch.tensor]
        N points in latent space (latent means). tensor shape: N x latent_dim

    The function returns a scalar.
    """
    pd = torch.cdist(curve_points, latent)  # M x N
    pd_min, _ = torch.min(pd, dim=1)
    pd_min_max = pd_min.max()
    return pd_min_max

# Fisher-Rao implmentation
def kl_divergence_between_decoders(decoders, z1, z2):
    # Get the distributions from each decoder
    if len(decoders) == 1:
        dist1 = decoders[0](z1)
        dist2 = decoders[0](z2)
    else:
        d1 = decoders[random.randint(0, len(decoders) - 1)]
        d2 = decoders[random.randint(0, len(decoders) - 1)]
        dist1 = d1(z1)
        dist2 = d2(z2)

    # ensure numerical stability
    epsilon = 1e-6
    clamped_probs1 = torch.clamp(dist1.base_dist.probs, epsilon, 1 - epsilon)
    clamped_probs2 = torch.clamp(dist2.base_dist.probs, epsilon, 1 - epsilon)
    
    base_dist1 = torch.distributions.ContinuousBernoulli(clamped_probs1)
    base_dist2 = torch.distributions.ContinuousBernoulli(clamped_probs2)
    
    new_dist1 = torch.distributions.Independent(base_dist1, 3)
    new_dist2 = torch.distributions.Independent(base_dist2, 3)

    # Compute the KL divergence between the two distributions
    kl_div = td.kl_divergence(new_dist1, new_dist2)
    
    if kl_div < 0:
        print(kl_div)

    return kl_div

# plot data and random geodesics
def plotgeodesics(decoders, latents, labels, curve_indices, num_curves, filename):
    
    ## Plot training data
    plt.figure()
    for k in range(num_classes):
        idx = labels == k
        plt.scatter(latents[idx, 0].cpu(), latents[idx, 1].cpu())

    ## Plot random geodesics

    geodesics = []
    with tqdm(total = num_curves) as pbar:
        for k in range(num_curves):
            z0 = latents[curve_indices[k, 0]]
            z1 = latents[curve_indices[k, 1]]
            num_points = 20
            t = torch.linspace(0, 1, num_points, device=device).reshape(num_points, 1)
            
            initial_curve = (1-t) * z0 + t * z1
            parameters = torch.nn.Parameter(initial_curve[1:-1])
            optimizer = torch.optim.Adam([parameters], lr=0.1)
            # z_norm = torch.norm(z0, dim=-1)[None].to(device)
            
            def closure():
                optimizer.zero_grad() 
                curve = torch.cat([z0[None, :], parameters, z1[None, :]], dim=0)
                energy = 0

                for i in range(num_points-1):

                    # Compute the KL divergence for the current pair of points
                    if len(decoders) == 1:
                        kl_div = kl_divergence_between_decoders(decoders, curve[i], curve[i + 1])
                    else: 
                        kl_divlist = []
                        for _ in range(5):
                            kl_divlist.append(kl_divergence_between_decoders(decoders, curve[i], curve[i + 1]))
                        kl_div = sum(kl_divlist)/len(kl_divlist)

                    energy += kl_div

                energy.backward()
                return energy

            for _ in range(50): # Decrease this value if it's too time-consuming.
                optimizer.step(closure)

            pbar.set_description(f"{filename}, k = {k+1}/{num_curves}")
            geodesics.append(torch.cat([z0[None, :], parameters.detach(), z1[None, :]], dim=0))
            pbar.update(1)


    for geo in geodesics:
        plt.plot(geo[:, 0].cpu(), geo[:, 1].cpu())
    
    geo_tensor = torch.stack(geodesics, dim=0)
    # print("geo_tensor.size(): ", geo_tensor.size())
    torch.save(geo_tensor, 'geo_'+filename+'.pt')

    plt.savefig(filename+'.png')
    return 

if __name__ == "__main__":
    from torchvision import datasets, transforms
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'plot', 'ensembletrain', 'plotproximity'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--plot', type=str, default='plot.png', help='file to save latent plot in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)
    
    num_train_data = 2048
    num_test_data = 16  # we keep this number low to only compute a few geodesics
    num_classes = 3
    train_tensors = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Define VAE model
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(new_decoder())
    model = VAE(prior, decoder, encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'ensembletrain':

        # Model Initialization
        modellist = []
        oplist = []
        for _ in range(10):
            decoder = BernoulliDecoder(new_decoder()) # Randomization here.
            model = VAE(prior, decoder, encoder).to(device)
            modellist.append(model)
            
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            oplist.append(optim)

        
        # Model Selection and Training
        num_steps = len(mnist_train_loader)*args.epochs
        epoch = 0

        with tqdm(range(num_steps)) as pbar:
            for step in pbar:
                # Data Selection and Randomization
                x = next(iter(mnist_train_loader))[0]
                x = noise(x.to(device))

                # Model Selection
                model_idx = int(10 * torch.rand(1).item())
                model = modellist[model_idx]
                model.train()

                # Training
                optim = oplist[model_idx]
                optim.zero_grad()
                loss = model(x)
                loss.backward()
                optim.step()

                # Broadcast Encoder and Update Models and Ops
                updated_encoder = model.getencoder()
                oplist[model_idx] = optim
                for idx in range(10):
                    updated_prior = modellist[idx].getprior()
                    updated_decoder = modellist[idx].getdecoder()
                    updated_model = VAE(updated_prior, updated_decoder, updated_encoder).to(device)
                    modellist[idx] = updated_model
                
                # Report
                if step % 5 ==0 :
                    loss = loss.detach().cpu()
                    pbar.set_description(f"model idx = {model_idx}, epoch={epoch}, step={step}, loss={loss:.1f}")

                if (step+1) % len(mnist_train_loader) == 0:
                    epoch += 1
        
        # Save models
        torch.save(modellist, 'ensemblemodels.pt')


    elif args.mode == 'plot':
        ## Load trained model in single VAE and Ensemble VAE
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        modellist = [ VAE(prior, decoder, encoder) ] * 10
        modellist = torch.load('ensemblemodels.pt')
        enmodel = modellist[0].to(device)
        enmodel.eval()

        ## Encode test and train data
        latents, enlatents, labels = [], [], []
        with torch.no_grad():
            for x, y in mnist_train_loader:
                z = model.encoder(x.to(device))
                enz = enmodel.getencoder()(x.to(device))
                latents.append(z.mean)
                enlatents.append(enz.mean)
                labels.append(y.to(device))
            latents = torch.concatenate(latents, dim=0)
            enlatents = torch.concatenate(enlatents, dim=0)
            labels = torch.concatenate(labels, dim=0)

            
        num_curves = 100
        curve_indices = torch.randint(num_train_data, (num_curves, 2), device=device)  # (num_curves) x 2
        
        ## Uncomment this to plot curves in Part A.
        # plotgeodesics(
        #     decoders=[model.decoder],
        #     latents=latents, 
        #     labels=labels, 
        #     curve_indices=curve_indices,
        #     num_curves=num_curves,
        #     filename='singleVAE')
        
        decoderlist = []
        for i in range(len(modellist)):
            decoderlist.append(modellist[i].getdecoder())

        plotgeodesics(
            decoders=decoderlist,
            latents=enlatents, 
            labels=labels, 
            curve_indices=curve_indices,
            num_curves=num_curves,
            filename='ensembleVAE')


    elif args.mode == 'plotproximity':
        modellist = [ VAE(prior, decoder, encoder) ] * 10
        modellist = torch.load('ensemblemodels.pt')
        enmodel = modellist[0].to(device)
        enmodel.eval()

        ## Encode test and train data
        enlatents, labels = [], []
        with torch.no_grad():
            for x, y in mnist_train_loader:
                enz = enmodel.getencoder()(x.to(device))
                enlatents.append(enz.mean)
                labels.append(y.to(device))
            enlatents = torch.concatenate(enlatents, dim=0)
            labels = torch.concatenate(labels, dim=0)

        num_curves = 50
        curve_indices = torch.randint(num_train_data, (num_curves, 2), device=device)  # (num_curves) x 2

        decoderlist = []
        for i in range(len(modellist)):
            decoderlist.append(modellist[i].getdecoder())

        # Start plotting
        ave_prox_list = []
        for i in range(len(decoderlist)):
            dl = (decoderlist[:i+1])
            plotgeodesics(
                decoders=dl,
                latents=enlatents, 
                labels=labels, 
                curve_indices=curve_indices,
                num_curves=num_curves,
                filename='ensembleVAE{}'.format(i+1))
            
            geo_tensor = torch.load('geo_'+'ensembleVAE{}'.format(i+1)+'.pt')
            proximity_list = []
            for i in range(geo_tensor.size()[0]):
                proximity_list.append(proximity(geo_tensor[1], enlatents))

            ave_prox = (sum(proximity_list)/len(proximity_list)).item()
            print("ave_prox", ave_prox)
            ave_prox_list.append(ave_prox)

        numlist = list(range(1, len(ave_prox_list) + 1))
        
        plt.figure()
        plt.plot(numlist,ave_prox_list,)
        plt.xlabel("Number of ensemble members")
        plt.ylabel("Average proximity")
        plt.savefig("AveProximity.png")



