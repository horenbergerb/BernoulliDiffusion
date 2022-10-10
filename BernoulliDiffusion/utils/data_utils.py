import torch

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data_from_file(data_dir):
    data = []
    with open(os.path.join(data_dir), 'r') as f:
        for line in f:
            data.append(torch.FloatTensor([float(x) for x in line.strip()]).to(device))

    data = torch.vstack(data)
    return data