
import os

from BernoulliDiffusion.utils.data_utils import load_data_from_file

class Validator:
    '''Quantifies the performance of the model by generating a large batch of samples
    via the reverse process and measuring what proportion lie within training, validation,
    or neither dataset'''
    def __init__(self, working_dir, cfg):
        self.working_dir = working_dir
        self.cfg = cfg

        # sets make membership checks constant time; lists took ~10ms per sample on large datasets
        self.train_data = self.load_as_set_(os.path.join(self.working_dir, 'train.txt'))
        self.val_data = self.load_as_set_(os.path.join(self.working_dir, 'val.txt'))

    def load_as_set_(self, data_dir):
        return {tuple(x) for x in load_data_from_file(data_dir).cpu().detach().numpy().tolist()}

    def validate(self, model, num_samples, batch_size):
        train_count = 0
        val_count = 0
        other_count = 0

        sample_count = 0
        while sample_count < num_samples:
            samples = model.p_sample(batch_size).cpu().detach().numpy().tolist()

            for sample in samples:
                sample = tuple(sample)
                if sample in self.train_data:
                    train_count += 1
                elif sample in self.val_data:
                    val_count += 1
                else:
                    other_count += 1
                sample_count += 1
                if sample_count >= num_samples:
                    break

        return train_count, val_count, other_count

