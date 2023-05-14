import pandas as pd
from src.tasks.metrics import get_reg_metrics
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def save_feature_importance(df, num_features):
    if df is not None:
        df.sort_values(['importance'], ascending=[False], inplace=True)
        df['importance'] = df['importance'] / df['importance'].sum()
        df_fig = df.iloc[0:num_features, :]
        plt.figure(figsize=(8, 0.3 * df_fig.shape[0]))
        sns.set_theme(style='whitegrid', font_scale=1)
        bar = sns.barplot(
            data=df_fig,
            y='feature_label',
            x='importance',
            edgecolor='black',
            orient='h',
            dodge=True
        )
        bar.set_xlabel("Importance")
        bar.set_ylabel("")
        plt.savefig(f"feature_importance.png", bbox_inches='tight', dpi=400)
        plt.savefig(f"feature_importance.pdf", bbox_inches='tight')
        plt.close()
        df.set_index('feature', inplace=True)
        df.to_excel("feature_importance.xlsx", index=True)

def eval_regression(config, y_real, y_pred, loggers, part, is_log=True, is_save=True, file_suffix=''):
    metrics = get_reg_metrics()

    if is_log:
        if 'wandb' in config.logger:
            for m in metrics:
                wandb.define_metric(f"{part}/{m}", summary=metrics[m][1])

    metrics_df = pd.DataFrame(index=[m for m in metrics], columns=[part])
    metrics_df.index.name = 'metric'
    log_dict = {}
    for m in metrics:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_torch = torch.from_numpy(y_pred)
        m_val = float(metrics[m][0](y_pred_torch, y_real_torch).numpy())
        metrics[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val

    if loggers is not None:
        for logger in loggers:
            if is_log:
                logger.log_metrics(log_dict)

    if is_save:
        metrics_df.to_excel(f"metrics_{part}{file_suffix}.xlsx", index=True)

    return metrics_df


def eval_loss(loss_info, loggers, is_log=True, is_save=True, file_suffix=''):
    for epoch_id, epoch in enumerate(loss_info['epoch']):
        log_dict = {
            'epoch': loss_info['epoch'][epoch_id],
            'trn/loss': loss_info['trn/loss'][epoch_id],
            'val/loss': loss_info['val/loss'][epoch_id]
        }
        if loggers is not None:
            for logger in loggers:
                if is_log:
                    logger.log_metrics(log_dict)

    if is_save:
        loss_df = pd.DataFrame(loss_info)
        loss_df.set_index('epoch', inplace=True)
        loss_df.to_excel(f"loss{file_suffix}.xlsx", index=True)
