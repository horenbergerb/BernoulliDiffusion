import torch

import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please provide a model and a number of samples to generate, for example:\n`python {} runs/heartbeat_len_20_per_5/results/d_10_11_2022_t_17_20/checkpoints/model_epoch_29.pt` 100'.format(sys.argv[0]))
        exit()

    diffusion_model = torch.load(sys.argv[1])

    samples = diffusion_model.p_sample(int(sys.argv[2])).int().tolist()
    for sample in samples:
        print(''.join([str(x) for x in sample]))