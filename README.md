# Bernoulli Diffusion

## About

This is an implementation of the diffusion machine learning algorithm described in [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585). Notably, this implementation is designed to train on binary-valued data. It uses a Bernoulli distribution to generate noise instead of a Gaussian.

The model in `model.py` reflects the architecture described in the original paper. The training loop is stored in `train.py`.

The tools in `data.py` generate "heartbeat data," periodic data with a random shift. Here are two examples of heartbeat data with period 5 and sequence_length 20:
```
    [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]
    [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0]
```
You can train the model to generate samples of heartbeat data.

`main.py` is the heart of this repository and initializes training based on the configuration in `config.yaml`.

Finally, there are various unit tests in `test_unittests.py`.

## Running this code

Make sure to create a Python3 environment and install the requirements in `requirements.txt`.

You can train a diffusion model on heartbeat data by simply executing the following command from the root directory of the project:

```
python main.py
```

You can configure the session by editing the settings in `config.yaml`

You can run the unit tests from the root directory with the following command:

```
# run all unit tests
python -m unittest discover BernoulliDiffusion/tests/

# run a particular file of unit tests
python -m unittest BernoulliDiffusion/tests/test_filename.py
```

## Mathematics and Derivations

I wrote a whole blog post about the derivations which you can find [here](https://horenbergerb.github.io/2022/10/03/bernoulliderivations.html).

It contains notes on all of the trickier topics that I encountered while writing this code.