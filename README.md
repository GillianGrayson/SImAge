

# Small Immuno Age (SImAge)

[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

Repository with source code for paper "Deep Learning and Gradient Boosting for small immunological clocks" by 
[A. Kalyakulina](https://orcid.org/0000-0001-9277-502X),
[I. Yusipov](http://orcid.org/0000-0002-0540-9281),
[E. Kondakova](http://orcid.org/0000-0002-6123-8181),
[M.G. Bacalini](http://orcid.org/0000-0003-1618-2673),
[C. Franceschi](http://orcid.org/0000-0001-9841-6386),
[M. Vedunova](http://orcid.org/0000-0001-9759-6477),
[M. Ivanchenko](http://orcid.org/0000-0002-1903-7423).

## Description 

Данный репозиторий содержит коды для построения и анализа различных моделей машинного обучения, используемых для решения задачи регрессии хронологического возраста по табличным  данным иммунологического профиля.

### Основные возможности:
- Построение различных моделей машинного обучения для табличных данных (Linear, GBDT, and DNN).
- Hyperparameter search для моделей машинного обучения.
- Интерпретируемости и применение методов объяснимого искусственного интеллекта (XAI).

### Main Technologies

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research.
- [Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

## Project Structure

The directory structure is following:

```
├── configs                <- Hydra configuration files
│   ├── callbacks             <- Callbacks configs
│   ├── datamodule            <- Datamodule configs
│   ├── experiment            <- Experiment configs
│   ├── hparams_search        <- Hyperparameter search configs
│   ├── hydra                 <- Hydra configs
│   ├── logger                <- Logger configs
│   ├── model                 <- Model configs
│   ├── trainer               <- Trainer configs
│   └── main.yaml             <- Main configs
│
├── data                   <- Immunological data and generated results
│
├── src                    <- Source code
│   ├── datamodules           <- Datamodules
│   ├── models                <- Models
│   └── utils                 <- Utility scripts
│
├── requirements.txt       <- File for installing python dependencies
├── run.py                 <- Main run file
└── README.md
```

## Install dependencies

```bash
# clone project
git clone https://github.com/GillianGrayson/SImAge
cd SImAge

# [OPTIONAL] create conda environment
conda create -n env_name python=3.9
conda activate env_name

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Data description
Данные иммунологического профиля представляют из себя значения концентраций 46 цитокинов в плазме крови для 260 человек.
В контексте этих данных решается задача регрессии хронологического возраста.

### Файловая структура:
```
└── data                    <- Эксперимент по регрессии возраста
   ├── models                 <- Результаты экспериментов для разных моделей
   ├── data.xlsx              <- Dataframe with immunological data
   ├── feats_con_46.xlsx      <- File with all 46 biomarkers
   └── feats_con_10.xlsx      <- File with the most imortant 10 biomarkers
```
> `data.xlsx` is a dataframe, each row corresponds to sample, each column corresponds to feature.
> In addition to immunological features there are also `Age` (in years) and `Sex` (F or M).

> `feats_con_*.xlsx` are dataframes which contains features (immunological biomarkers), which will be used as input features of models.

> `models` - это директория, в которой будут сохраняться результаты разных моделей (logs, figures, tables).


## Configuring experiments

### Main config

Location: [configs/main.yaml](configs/main.yaml) <br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python main.py`.<br>

<details>
<summary><b>Main project config details</b></summary>

```yaml
# order of defaults determines the order in which configs override each other
defaults:
  - _self_
  - experiment: train     # Global parameters of experiment
  - datamodule: tabular   # Information about dataset 
  - trainer: gpu          # Run configuration for DNN models
  - callbacks: default    # Callbacks for DNN models
  - logger: none          # Loggers for DNN models
  - hydra: default.yaml   # Output paths for logs
  - model: danet          # Model
  - hparams_search: null  # Model-specific hyperparameters

  # enable color logging
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog

```
</details>

### Experiment config

Location: [configs/experiment](configs/experiment)<br>
Experiment config contains global parameters.<br>
Executing command: `python main.py experiment=train`.<br>

<details>
<summary><b>Experiment config details</b></summary>

```yaml
# @package _global_

# Global params
seed: 1337          # Random seed
task: "regression"  # Task type
target: "Age"       # Target column name

# Cross-validation params
cv_is_split: True   # Perform cross-validation?
cv_n_splits: 5      # Number of splits in cross-validation
cv_n_repeats: 5     # Number of repeats in cross-validation

# Data params
in_dim: 10      # Number of input features
out_dim: 1      # Output dimension
embed_dim: 16   # Default embedding dimension

# Optimization metrics params
optimized_metric: "mean_absolute_error"   # All metrics listed in src.tasks.metrics
optimized_mean: "cv_mean"                 # Optimize mean result across all cross-validation splits? Options: ["", "cv_mean"]
optimized_part: "val"                     # Optimized data partition. Options: ["val", "tst"]
direction: "min"                          # Direction of metrics optimization. Options ["min", "max"]

# Run params
max_epochs: 1000          # Maximum number of epochs
patience: 100             # Number of early stopping epochs
feature_importance: none  # Feature importance method. Options: [none, shap_deep, shap_kernel, shap_tree, native]

# Info params
debug: False                # Is Debug?
print_config: False         # Print config?
print_model: False          # Print model info?
ignore_warnings: True       # Ignore warnings?
test_after_training: True   # Test after training?

# Directories and files params
project_name: ${model.name}
base_dir: "${oc.env:PROJECT_ROOT}/data"
data_dir: "${base_dir}"
work_dir: "${base_dir}/models/${project_name}"

# SHAP values params
is_shap: False                      # Calculate SHAP values?
is_shap_save: False                 # Save SHAP values?
shap_explainer: "Tree"              # Type of explainer. Options: ["Tree", "Kernel", "Deep"]
shap_bkgrd: "tree_path_dependent"   # Type of background data. Options: ["trn", "all", "tree_path_dependent"]

# Plot params
num_top_features: 10  # Number of most important features to plot
num_examples: 10      # Number of samples to plot some SHAP figures
```
</details>

### Datamodule config

Location: [configs/datamodule](configs/datamodule)<br>
Datamodule config contains information about loaded dataset, input and target features.<br>
Executing command: `python main.py datamodule=tabular`.<br>

<details>
<summary><b>Datamodule config details</b></summary>

```yaml
_target_: src.datamodules.tabular.TabularDataModule   # Instantiated object
task: "regression"                                    # Task type. Options: ["classification", "regression"]. Here we solve regression problem
feats_con_fn: "${data_dir}/feats_con_${in_dim}.xlsx"  # File with continuous input features
feats_cat_fn: null                                    # File with categorical input features
feats_cat_encoding: label                             # How to encode categorical features? Options: ["label", "one_hot"]
feats_cat_embed_dim: ${embed_dim}                     # Dimension size for categorical features embedding
target: ${target}                                     # Target predicted feature
target_classes_fn: null                               # File with selected classes (for classification tasks only)
data_fn: "${data_dir}/data.xlsx"                      # File with dataset
data_index: index                                     # Index column in dataset file
data_imputation: fast_knn                             # Imputation method for missing values (see https://github.com/eltonlaw/impyute)
split_by: trn_val                                     # Splitting method. Options: [trn_val, top_feat, explicit_feat]
split_trn_val: [0.80, 0.20]                           # Splitting parts for "trn_val" splitting method
split_top_feat: null                                  # Splitting column for "top_feat" splitting method
split_explicit_feat: Split                            # Splitting column for "explicit_feat" splitting method
batch_size: 512                                       # Batch size (for torch DataLoader)
num_workers: 0                                        # Num workers (for torch DataLoader)
pin_memory: False                                     # Memory pinning (for torch DataLoader)
seed: ${seed}                                         # Random seed
weighted_sampler: True                                # Samplers are wighted? For imbalanced data
```
</details>

### Trainer config

Location: [configs/trainer](configs/trainer)<br>
Trainer config contains information about different aspects of DNN training process.<br>
Данный конфигурационынй файл используется для инициализации [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).<br>
Executing command: `python main.py trainer=cpu`.<br>

<details>
<summary><b>Trainer config example</b></summary>

```yaml
# All Trainer parameters available here:
# [https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html]
_target_: pytorch_lightning.Trainer   # Instantiated object
gpus: 0                               # Number of GPUs to train on
min_epochs: 1                         # Force training for at least these many epochs
max_epochs: ${max_epochs}             # Stop training once this number of epochs is reached
```
</details>

## Примеры запусков

<details>
<summary><b>Override any config parameter from command line</b></summary>

```bash
# change the maximum number of epochs
python run.py max_epochs=200
```
</details>

<details>
<summary><b>Поддерживается тренировка нейросетевых моделей как на CPU, так и на GPU</b></summary>

```bash
# train on CPU
python run.py trainer=cpu

# train on GPU
python run.py trainer=gpu
```
</details>


<details>
<summary><b>Train neural network models with any logger available in PyTorch Lightning (e.g. W&B)</b></summary>

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python run.py logger=wandb
```
> **Note**: Using wandb requires you to [setup account](https://www.wandb.com/) first.
</details>


<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of learning rate and min data in leaf in LightGBM model
python run.py --multirun model=lightgbm model.learning_rate=0.1,0.05,0.01 model.min_data_in_leaf=5,10
```
</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/lightgbm.yaml`
# over chosen experiment config for LightGBM model as example
python run.py --multirun model=lightgbm hparams_search=lightgbm
```
> **Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes).
</details>



## Workflow

**Basic workflow**

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [datamodules/mnist_datamodule.py](src/datamodules/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config:
   ```bash
   python src/train.py experiment=experiment_name.yaml
   ```

**Experiment design**

_Say you want to execute many runs to plot how accuracy changes in respect to batch size._

1. Execute the runs with some config parameter that allows you to identify them easily, like tags:

   ```bash
   python train.py -m logger=csv datamodule.batch_size=16,32,64,128 tags=["batch_size_exp"]
   ```

2. Write a script or notebook that searches over the `logs/` folder and retrieves csv logs from runs containing given tags in config. Plot the results.

<br>

## Logs

Hydra creates new output directory for every executed run.

Default logging structure:

```
├── logs
│   ├── task_name
│   │   ├── runs                        # Logs generated by single runs
│   │   │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
│   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   ├── csv                     # Csv logs
│   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   └── ...
│   │   │
│   │   └── multiruns                   # Logs generated by multiruns
│   │       ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
│   │       │   ├──1                        # Multirun job number
│   │       │   ├──2
│   │       │   └── ...
│   │       └── ...
│   │
│   └── debugs                          # Logs generated when debugging config is attached
│       └── ...
```


You can change this structure by modifying paths in [hydra configuration](configs/hydra).


## Experiment Tracking

PyTorch Lightning supports many popular logging frameworks: [Weights&Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLFlow](https://mlflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/).

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:

```bash
python train.py logger=logger_name
```

You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).

You can also write your own logger.

Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the [docs](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [MNIST example](src/models/mnist_module.py).


## Hyperparameter Search

You can define hyperparameter search by adding new config file to [configs/hparams_search](configs/hparams_search).

<details>
<summary><b>Show example hyperparameter search config</b></summary>

```yaml
# @package _global_

defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
# make sure this is the correct name of some metric logged in lightning module!
optimized_metric: "val/acc_best"

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      datamodule.batch_size: choice(32, 64, 128, 256)
      model.net.lin1_size: choice(64, 128, 256)
      model.net.lin2_size: choice(64, 128, 256)
      model.net.lin3_size: choice(32, 64, 128, 256)
```

</details>

Next, execute it with: `python train.py -m hparams_search=mnist_optuna`

Using this approach doesn't require adding any boilerplate to code, everything is defined in a single config file. The only necessary thing is to return the optimized metric value from the launch file.

You can use different optimization frameworks integrated with Hydra, like [Optuna, Ax or Nevergrad](https://hydra.cc/docs/plugins/optuna_sweeper/).

The `optimization_results.yaml` will be available under `logs/task_name/multirun` folder.

This approach doesn't support advanced techniques like prunning - for more sophisticated search, you should probably write a dedicated optimization task (without multirun feature).


## License

This repository is licensed under the MIT License.
