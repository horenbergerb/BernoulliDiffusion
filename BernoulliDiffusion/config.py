from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    T: int
    num_sample_steps: int
    hidden_size: int = 50
    num_hidden_layers: int = 2
    predict_x0: bool = False
    x0_loss_weight: float = 0.0


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    num_examples: int
    num_val_samples: int
    val_batch_size: int
    save_every_n_epochs: int
    clip_thresh: float
    validate_every_n_epochs: int = 100
    examples_every_n_epochs: int = 10


@dataclass
class DataConfig:
    batch_size: int


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