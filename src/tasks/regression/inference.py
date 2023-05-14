import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.datamodules.tabular import TabularDataModule
from src.utils import utils
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from src.tasks.routines import eval_regression
from catboost import CatBoost
import plotly.express as px
from src.tasks.regression.shap import explain_shap
from src.models.tabular.base import get_model_framework_dict
import pickle


log = utils.get_logger(__name__)

def inference_regression(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    model_framework_dict = get_model_framework_dict()
    model_framework = model_framework_dict[config.model.name]

    # Init Lightning datamodule for test
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    features = datamodule.get_features()
    num_features = len(features['all'])
    config.in_dim = num_features
    target = datamodule.target
    target_label = datamodule.target_label
    df = datamodule.get_data()

    df = df[df[config.data_part_column].notna()]
    data_parts = df[config.data_part_column].dropna().unique()
    data_part_main = config.data_part_main
    data_parts = [data_part_main] + list(set(data_parts) - set([data_part_main]))
    indexes = {}
    X = {}
    y = {}
    y_pred = {}
    colors = {}
    for data_part_id, data_part in enumerate(data_parts):
        indexes[data_part] = df.loc[df[config.data_part_column] == data_part, :].index.values
        X[data_part] = df.loc[indexes[data_part], features['all']].values
        y[data_part] = df.loc[indexes[data_part], target].values
        colors[data_part] = px.colors.qualitative.Light24[data_part_id]

    if model_framework == "pytorch":
        widedeep = datamodule.get_widedeep()
        embedding_dims = [(x[1], x[2]) for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
        categorical_cardinality = [x[1] for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
        if config.model.name.startswith('widedeep'):
            config.model.column_idx = widedeep['column_idx']
            config.model.cat_embed_input = widedeep['cat_embed_input']
            config.model.continuous_cols = widedeep['continuous_cols']
        elif config.model.name.startswith('pytorch_tabular'):
            config.model.continuous_cols = features['con']
            config.model.categorical_cols = features['cat']
            config.model.embedding_dims = embedding_dims
            config.model.categorical_cardinality = categorical_cardinality
        elif config.model.name == 'nam':
            num_unique_vals = [len(np.unique(X[data_part_main][:, i])) for i in range(X[data_part_main].shape[1])]
            num_units = [min(config.model.num_basis_functions, i * config.model.units_multiplier) for i in num_unique_vals]
            config.model.num_units = num_units
        log.info(f"Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)

        model = type(model).load_from_checkpoint(checkpoint_path=f"{config.path_ckpt}")
        model.eval()
        model.freeze()

        for data_part in data_parts:
            y_pred[data_part] = model(torch.from_numpy(X[data_part])).cpu().detach().numpy().ravel()

        def predict_func(X):
            batch = {
                'all': torch.from_numpy(np.float32(X[:, features['all_ids']])),
                'continuous': torch.from_numpy(np.float32(X[:, features['con_ids']])),
                'categorical': torch.from_numpy(np.int32(X[:, features['cat_ids']])),
            }
            tmp = model(batch)
            return tmp.cpu().detach().numpy()

    elif model_framework == "stand_alone":
        if config.model.name == "xgboost":
            model = xgb.Booster()
            model.load_model(config.path_ckpt)

            for data_part in data_parts:
                dmat = xgb.DMatrix(X[data_part], y[data_part], feature_names=features['all'], enable_categorical=True)
                y_pred[data_part] = model.predict(dmat)

            def predict_func(X):
                X = xgb.DMatrix(X, feature_names=features['all'], enable_categorical=True)
                y = model.predict(X)
                return y

        elif config.model.name == "catboost":
            model = CatBoost()
            model.load_model(config.path_ckpt)

            for data_part in data_parts:
                y_pred[data_part] = model.predict(X[data_part]).astype('float32')

            def predict_func(X):
                X = pd.DataFrame(data=X, columns=features["all"])
                X[features["cat"]] = X[features["cat"]].astype('int32')
                y = model.predict(X)
                return y

        elif config.model.name == "lightgbm":
            model = lgb.Booster(model_file=config.path_ckpt)

            for data_part in data_parts:
                y_pred[data_part] = model.predict(X[data_part], num_iteration=model.best_iteration).astype('float32')

            def predict_func(X):
                y = model.predict(X, num_iteration=model.best_iteration)
                return y

        elif config.model.name == "elastic_net":
            model = pickle.load(open(config.path_ckpt, 'rb'))

            for data_part in data_parts:
                y_pred[data_part] = model.predict(X[data_part]).astype('float32')
            def predict_func(X):
                y = model.predict(X)
                return y

        else:
            raise ValueError(f"Model {config.model.name} is not supported")

    else:
        raise ValueError(f"Unsupported model_framework: {model_framework}")

    for data_part in data_parts:
        df.loc[indexes[data_part], "Prediction"] = y_pred[data_part]
        eval_regression(config, y[data_part], y_pred[data_part], None, data_part, is_log=False, is_save=True, file_suffix=f"")
    df["Prediction error"] = df['Prediction'] - df[f"{target}"]
    df["Prediction error abs"] = df["Prediction error"].abs()

    df_fig = df.loc[:, [target, 'Prediction', "Prediction error"]].copy()
    for data_part in data_parts:
        df_fig.loc[indexes[data_part], 'Part'] = data_part
    plt.figure()
    sns.set_theme(style='whitegrid')
    xy_min = df_fig[[target, 'Prediction']].min().min()
    xy_max = df_fig[[target, 'Prediction']].max().max()
    xy_ptp = xy_max - xy_min
    scatter = sns.scatterplot(
        data=df_fig,
        x=target,
        y="Prediction",
        hue="Part",
        palette=colors,
        linewidth=0.3,
        alpha=0.75,
        edgecolor="k",
        s=25,
        hue_order=list(colors.keys())
    )
    scatter.set_xlabel(target_label)
    scatter.set_ylabel("Prediction")
    scatter.set_xlim(xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp)
    scatter.set_ylim(xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp)
    plt.gca().plot(
        [xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp],
        [xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp],
        color='k',
        linestyle='dashed',
        linewidth=1
    )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"scatter.png", bbox_inches='tight', dpi=400)
    plt.savefig(f"scatter.pdf", bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.set_theme(style='whitegrid')
    violin = sns.violinplot(
        data=df_fig,
        x="Part",
        y='Prediction error',
        palette=colors,
        scale='width',
        order=list(colors.keys()),
        saturation=0.75,
    )
    plt.savefig(f"violin.png", bbox_inches='tight', dpi=400)
    plt.savefig(f"violin.pdf", bbox_inches='tight')
    plt.close()

    df['ids'] = np.arange(df.shape[0])
    ids = {}
    for data_part in data_parts:
        ids[data_part] = df.loc[indexes[data_part], 'ids'].values
    ids['all'] = df['ids']

    expl_data = {
        'model': model,
        'predict_func': predict_func,
        'df': df,
        'features': features,
        'target': target,
        'ids': ids
    }
    if config.is_shap == True:
        explain_shap(config, expl_data)

    df.to_excel("df.xlsx", index=True)
