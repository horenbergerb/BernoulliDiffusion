from dataclasses import dataclass
import yaml

@dataclass
class Config:
    sequence_length: int
    period: int
    T: int
    batch_size: int
    num_batches: int
    num_sample_steps: int
    epochs: int
    lr: float
    training_info_freq: int
    save_every_n_epochs: int
    output_dir: str


def load_config(cfg_dir: str) -> Config:
    cfg_dict = None
    with open(cfg_dir, "r") as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    return Config(**cfg_dict)