# Bernoulli Diffusion

## About

This is an implementation of the diffusion machine learning algorithm described in [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585). It trains a diffusion model on heartbeat data.

The tools in `data.py` generate "heartbeat data," periodic data with a random shift. Here are two examples of heartbeat data with period 5 and sequence_length 20:
```
    [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]
    [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0]
```

The model in `model.py` reflects the architecture described in the original paper. The training loop is stored in `train.py`.

`main.py` is the heart of this repository and initializes training based on the configuration in `config.yaml`.

Finally, there are various unit tests in `test_unittests.py`.

## Running this code

Make sure to create a Python3 environment and install the requirements in `requirements.txt`.

You can train a diffusion model on heartbeat data by simply executing the following command from the root directory of the project:

```
python src/main.py
```

You can configure the session by editing the settings in `config.yaml`