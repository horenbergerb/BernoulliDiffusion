
import os

from BernoulliDiffusion.utils.data_utils import load_data_from_file

class Validator:
    def __init__(self, cfg):
        self.cfg = cfg

    def validate(self, model, num_samples, batch_size):
        train_data = load_data_from_file(os.path.join(self.cfg.data_dir, 'train.txt')).cpu().detach().numpy().tolist()
        val_data = load_data_from_file(os.path.join(self.cfg.data_dir, 'val.txt')).cpu().detach().numpy().tolist()

        train_count = 0
        val_count = 0
        other_count = 0

        sample_count = 0
        while sample_count < num_samples:
            samples = model.p_sample(batch_size).cpu().detach().numpy().tolist()

            for sample in samples:
                if sample in train_data:
                    train_count += 1
                elif sample in val_data:
                    val_count += 1
                else:
                    other_count += 1
                sample_count += 1
                if sample_count >= num_samples:
                    break

        return train_count, val_count, other_count


