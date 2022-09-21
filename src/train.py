import torch
import torch.optim as optim
from torch.distributions import Binomial

from config import Config
from model import BinomialDiffusion
from data import generate_batch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_diffusion_model(cfg: Config,
                          diffusion_model: BinomialDiffusion) -> BinomialDiffusion:
    
    optimizer = optim.SGD(diffusion_model.parameters(), lr=cfg.lr)

    # Zero grad, input a random tensor, then backprop
    diffusion_model.zero_grad()
    m = Binomial(1, torch.zeros((cfg.batch_size, cfg.sequence_length)).fill_(0.5))
    input = m.sample().to(device)
    out = diffusion_model(input)
    diffusion_model.zero_grad()
    out.sum().backward()

    # load the training data, i.e. batches of heartbeat data
    batches = [generate_batch(num_samples=cfg.batch_size,
                              period=cfg.period,
                              sequence_length=cfg.sequence_length).to(device=device)
                              for i in range(0, cfg.num_batches)]

    # training loop
    losses = []
    for i in range(0, cfg.epochs):
        avg_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            output = diffusion_model(batch)
            output.backward()
            print('Loss: {}'.format(output.sum()))
            optimizer.step()
            avg_loss += output.sum()
        avg_loss = avg_loss / (cfg.batch_size * cfg.num_batches)
        losses.append(avg_loss)
        if i%cfg.training_info_freq == 0:
            print('Epoch: {}'.format(i))
            print('Losses per epoch: {}'.format(losses))
            print('Sample from reverse trajectory:')
            post_sample = diffusion_model.p_sample(cfg.batch_size)
            print(post_sample)
            print('Percent of sample digits which are 1: {}'.format(post_sample.sum()/torch.numel(post_sample)))

    return diffusion_model