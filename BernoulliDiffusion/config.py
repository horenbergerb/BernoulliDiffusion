from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    T: int
    num_sample_steps: int


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    training_info_freq: int
    num_examples: int
    num_val_samples: int
    val_batch_size: int
    save_every_n_epochs: int
    output_dir: str


@dataclass
class DataConfig:
    data_dir: str
    batch_size: int


def load_validation_config(cfg_dir):
    cfg_dict = None
    with open(cfg_dir, "r") as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
            if 'validation' not in cfg_dict:
                return None
            val_cfg = DataConfig(**cfg_dict['validation'])
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    return val_cfg


def load_config(cfg_dir: str):
    cfg_dict = None
    with open(cfg_dir, "r") as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
            model_cfg = ModelConfig(**cfg_dict['model'])
            training_cfg = TrainingConfig(**cfg_dict['training'])
            data_cfg = DataConfig(**cfg_dict['data'])
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    return model_cfg, training_cfg, data_cfg