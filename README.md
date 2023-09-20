# Summary

This is a brief introduction on how to use ar.py for a simple DeepAR training and evaluation with Pytorch Forecasting.

PyTorch Forecasting aims to ease state-of-the-art timeseries forecasting with neural networks for real-world cases and research alike. The goal is to provide a high-level API with maximum flexibility for professionals and reasonable defaults for beginners.
Specifically, the package provides

- A timeseries dataset class which abstracts handling variable transformations, missing values,
  randomized subsampling, multiple history lengths, etc.
- A base model class which provides basic training of timeseries models along with logging in tensorboard
  and generic visualizations such actual vs predictions and dependency plots
- Multiple neural network architectures for timeseries forecasting that have been enhanced
  for real-world deployment and come with in-built interpretation capabilities
- Multi-horizon timeseries metrics
- Hyperparameter tuning with [optuna](https://optuna.readthedocs.io/)

The package is built on [pytorch-lightning](https://pytorch-lightning.readthedocs.io/) to allow training on CPUs, single and multiple GPUs out-of-the-box.

# Installation

It is highly recommended to run this script inside a docker. Please use provided Dockerfile to generate docker image. The steps are as follows:<br>

1. Navigate to the folder where the dockerfile is located. Make sure requirements are inside the same folder
2. `docker build -t transformer_image .`
3. `docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/time_series_transformer/:/workspace/project/time_series_transformer --rm transformer_image`<br>

Then you are free to go inside docker container to run the scripts.

# Documentation

Visit [https://pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io) to read the
documentation with detailed tutorials.

# Methods

The Pytorch Forecasting library provides a [comparison of available models](https://pytorch-forecasting.readthedocs.io/en/latest/models.html).

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
  which outperforms DeepAR by Amazon by 36-69% in benchmarks
- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](http://arxiv.org/abs/1905.10437)
  which has (if used as ensemble) outperformed all other methods including ensembles of traditional statical
  methods in the M4 competition. The M4 competition is arguably the most important benchmark for univariate time series forecasting.
- [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](http://arxiv.org/abs/2201.12886) which supports covariates and has consistently beaten N-BEATS. It is also particularly well-suited for long-horizon forecasting.
- [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
  which is the one of the most popular forecasting algorithms and is often used as a baseline
- Simple standard networks for baselining: LSTM and GRU networks as well as a MLP on the decoder
- A baseline model that always predicts the latest known value

To implement new models or other custom components, see the [How to implement new models tutorial](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html). It covers basic as well as advanced architectures.

# Usage

```python
python3 ar.py
```

Networks can be trained with the [PyTorch Lighning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) on [pandas Dataframes](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe) which are first converted to a [TimeSeriesDataSet](https://pytorch-forecasting.readthedocs.io/en/latest/data.html).

# Todos

1. The current native-supporting logger is TensorBoardLogger in Pytorch Forecasting library, Wandb logger is not fully utilized. May replace it with TensorBoardLogger to have a better logging performance.
2. The hyper parameters searching are under investigation. May use PL own engine or [Optuna](https://optuna.readthedocs.io/) library if necessary.
3. Under current setting, it is obvious that the training is not converged.

# Miscellaneous

1. Under src folder, there are other files such as deepar.py/xt.py/util.py/time_series_transformers.ipynb. They are useless.
2. There are a train.py file in src folder as well. It has errors so cannot run it successfully, but it uses a raw time series transformer using HF. The detail can be found [here](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)
3. There is a **DeepAR investigation.pdf** to indicate some potential improvement directions. Also, there are other methods in the library to play. Having fun.
