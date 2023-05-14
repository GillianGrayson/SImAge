import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src.utils import utils
from slugify import slugify
import seaborn as sns


log = utils.get_logger(__name__)


def explain_samples(config, y_real, y_pred, indexes, shap_values, base_value, X, feature_names, features_labels, path):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    y_diff = np.array(y_pred) - np.array(y_real)
    order = np.argsort(y_diff)
    order_abs = np.argsort(np.abs(y_diff))
    num_examples = config.num_examples

    # Select samples with the biggest positive difference, the biggest positive difference, the smallest difference
    ids = list(set(np.concatenate((order[0:num_examples], order[-num_examples:], order_abs[0:num_examples]))))
    log.info(f"Number of samples: {len(ids)}")
    for m_id in ids:
        diff = y_diff[m_id]
        log.info(f"Plotting sample {m_id}: {indexes[m_id]} (real = {y_real[m_id]:0.4f}, pred = {y_pred[m_id]:0.4f}) with diff = {diff:0.4f}")

        ind_save = indexes[m_id]
        if isinstance(ind_save, str):
            ind_save = slugify(ind_save)

        Path(f"{path}/{ind_save}_real({y_real[m_id]:0.4f})_diff({diff:0.4f})").mkdir(parents=True, exist_ok=True)

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[m_id],
                base_values=base_value,
                data=X[m_id],
                feature_names=features_labels
            ),
            max_display=config.num_top_features,
            show=False,
        )
        fig = plt.gcf()
        plt.title(f"{indexes[m_id]}: Real = {y_real[m_id]:0.4f}, Pred = {y_pred[m_id]:0.4f}", {'fontsize': 20})
        fig.savefig(f"{path}/{ind_save}_real({y_real[m_id]:0.4f})_diff({diff:0.4f})/waterfall.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_real({y_real[m_id]:0.4f})_diff({diff:0.4f})/waterfall.png", bbox_inches='tight')
        plt.close()

        shap.plots.decision(
            base_value=base_value,
            shap_values=shap_values[m_id],
            features=X[m_id],
            feature_names=features_labels,
            show=False,
        )
        fig = plt.gcf()
        plt.title(f"{indexes[m_id]}: Real = {y_real[m_id]:0.4f}, Pred = {y_pred[m_id]:0.4f}", {'fontsize': 20})
        fig.savefig(f"{path}/{ind_save}_real({y_real[m_id]:0.4f})_diff({diff:0.4f})/decision.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_real({y_real[m_id]:0.4f})_diff({diff:0.4f})/decision.png", bbox_inches='tight')
        plt.close()

        shap.plots.force(
            base_value=base_value,
            shap_values=shap_values[m_id],
            features=X[m_id],
            feature_names=features_labels,
            show=False,
            matplotlib=True
        )
        fig = plt.gcf()
        fig.savefig(f"{path}/{ind_save}_real({y_real[m_id]:0.4f})_diff({diff:0.4f})/force.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_real({y_real[m_id]:0.4f})_diff({diff:0.4f})/force.png", bbox_inches='tight')
        plt.close()


def explain_shap(config, expl_data):
    model = expl_data['model']
    predict_func = expl_data['predict_func']
    df = expl_data['df']
    features_info = expl_data['features']
    features = features_info['all']
    features_labels = [features_info['labels'][f] for f in features_info['all']]
    target = expl_data['target']

    ids_bkgrd = expl_data["ids"][config.shap_bkgrd]
    indexes_bkgrd = df.index[ids_bkgrd]
    X_bkgrd = df.loc[indexes_bkgrd, features].values
    if config.shap_explainer == 'Tree':
        explainer = shap.TreeExplainer(model)
    elif config.shap_explainer == "Kernel":
        explainer = shap.KernelExplainer(predict_func, X_bkgrd)
    elif config.shap_explainer == "Deep":
        explainer = shap.DeepExplainer(model, torch.from_numpy(X_bkgrd))
    elif config.shap_explainer == "Sampling":
        explainer = shap.SamplingExplainer(predict_func, X_bkgrd)
    else:
        raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

    for part in expl_data["ids"]:
        print(f"part: {part}")
        if expl_data["ids"][part] is not None and len(expl_data["ids"][part]) > 0:
            log.info(f"Calculating SHAP for {part}")
            Path(f"shap/{part}/global").mkdir(parents=True, exist_ok=True)

            ids = expl_data["ids"][part]
            indexes = df.index[ids]
            X = df.loc[indexes, features].values
            y_pred = df.loc[indexes, "Prediction"].values

            if config.shap_explainer == "Tree":
                df_X = pd.DataFrame(data=X, columns=features_info["all"])
                df_X[features_info["cat"]] = df_X[features_info["cat"]].astype('int32')
                shap_values = explainer.shap_values(df_X)
                expected_value = explainer.expected_value
            elif config.shap_explainer in ["Kernel", "Sampling"]:
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[0]
            elif config.shap_explainer == "Deep":
                shap_values = explainer.shap_values(torch.from_numpy(X))
                expected_value = explainer.expected_value[0]
            else:
                raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

            if config.is_shap_save:
                df_shap = pd.DataFrame(index=indexes, columns=features, data=shap_values)
                df_shap.index.name = 'index'
                df_shap.to_excel(f"shap/{part}/shap.xlsx", index=True)
                df_expected_value = pd.DataFrame()
                df_expected_value.at["expected_value", part] = expected_value
                df_expected_value.to_excel(f"shap/{part}/expected_value.xlsx", index=True)

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=features_labels,
                max_display=config.num_top_features,
                plot_type="bar",
                show=False,
            )
            plt.savefig(f'shap/{part}/global/bar.png', bbox_inches='tight')
            plt.savefig(f'shap/{part}/global/bar.pdf', bbox_inches='tight')
            plt.close()

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=features_labels,
                max_display=config.num_top_features,
                plot_type="violin",
                show=False,
            )
            plt.savefig(f"shap/{part}/global/beeswarm.png", bbox_inches='tight')
            plt.savefig(f"shap/{part}/global/beeswarm.pdf", bbox_inches='tight')
            plt.close()

            explanation = shap.Explanation(
                values=shap_values,
                base_values=np.array([expected_value] * len(ids)),
                data=X,
                feature_names=features_labels
            )
            shap.plots.heatmap(
                explanation,
                show=False,
                max_display=config.num_top_features,
                instance_order=explanation.sum(1)
            )
            plt.savefig(f"shap/{part}/global/heatmap.png", bbox_inches='tight')
            plt.savefig(f"shap/{part}/global/heatmap.pdf", bbox_inches='tight')
            plt.close()

            Path(f"shap/{part}/features").mkdir(parents=True, exist_ok=True)
            mean_abs_impact = np.mean(np.abs(shap_values), axis=0)
            features_order = np.argsort(mean_abs_impact)[::-1]
            feat_ids_to_plot = features_order[0:config.num_examples]
            for rank, feat_id in enumerate(feat_ids_to_plot):
                feat = features[feat_id]
                # shap.dependence_plot(
                #     ind=feat_id,
                #     shap_values=shap_values,
                #     features=X,
                #     feature_names=features_labels,
                #     show=False,
                # )
                # plt.savefig(f"shap/{part}/features/{rank}_{feat}.png", bbox_inches='tight')
                # plt.savefig(f"shap/{part}/features/{rank}_{feat}.pdf", bbox_inches='tight')
                # plt.close()

                df_fig = pd.DataFrame({feat: X[:, feat_id], f"SHAP for {feat}": shap_values[:, feat_id], "Age": y_pred})
                scatter = sns.scatterplot(
                    data=df_fig,
                    x=feat,
                    y=f"SHAP for {feat}",
                    palette='spring',
                    hue="Age",
                    linewidth=0.2,
                    alpha=0.75,
                    edgecolor="k",
                )
                norm = plt.Normalize(df_fig['Age'].min(), df_fig['Age'].max())
                sm = plt.cm.ScalarMappable(cmap="spring", norm=norm)
                sm.set_array([])
                scatter.get_legend().remove()
                scatter.figure.colorbar(sm, label="Age")
                plt.savefig(f"shap/{part}/features/{rank}_{feat}.png", bbox_inches='tight')
                plt.savefig(f"shap/{part}/features/{rank}_{feat}.pdf", bbox_inches='tight')
                plt.close()

            explain_samples(
                config,
                df.loc[indexes, target].values,
                df.loc[indexes, "Prediction"].values,
                indexes,
                shap_values,
                expected_value,
                X,
                features,
                features_labels,
                f"shap/{part}/samples"
            )
