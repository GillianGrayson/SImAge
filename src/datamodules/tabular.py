import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
import numpy as np
import pandas as pd
from collections import Counter
from src.utils import utils
import plotly.express as px
import impyute.imputation.cs as imp
import pathlib
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import ast


log = utils.get_logger(__name__)

class TabularDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            feats_con: List[str] = None,
            feats_cat: List[str] = None,
            target: str = None
    ):
        self.feats_con = feats_con if feats_con else []
        self.feats_cat = feats_cat if feats_cat else []
        self.X_con = data.loc[:, self.feats_con].values
        self.X_cat = data.loc[:, self.feats_cat].values
        self.num_samples = data.shape[0]
        self.y = data.loc[:, target].values

    def __getitem__(self, idx: int):
        item = {
            "target": self.y[idx],
            "continuous": self.X_con[idx] if self.feats_con else torch.Tensor(),
            "categorical": self.X_cat[idx] if self.feats_cat else torch.Tensor(),
            "index": idx
        }
        item["all"] = np.concatenate((item["continuous"], item["categorical"]), axis=0, dtype=np.float32)
        return item

    def __len__(self):
        return self.num_samples


class TabularDataModule(LightningDataModule):

    def __init__(
            self,
            task: str = None,
            feats_con_fn: str = None,
            feats_cat_fn: str = None,
            feats_labels_col: str = None,
            feats_cat_replace_col: str = None,
            feats_cat_colors: str = None,
            feats_cat_encoding: str = None,
            feats_cat_keep: bool = True,
            feats_cat_embed_dim: int = None,
            target: str = None,
            target_label: str = None,
            target_classes_fn: str = None,
            duration: str = None,
            duration_label: str = None,
            data_fn: str = None,
            data_index: str = None,
            data_imputation: str = None,
            split_by: str = None,
            split_trn_val: Tuple[float, float] = (0.8, 0.2),
            split_top_feat: str = None,
            split_explicit_feat: str = None,
            batch_size: int = 4096,
            num_workers: int = 1,
            pin_memory: bool = False,
            seed: int = 42,
            weighted_sampler: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.task = task

        self.feats_con_fn = feats_con_fn
        self.feats_cat_fn = feats_cat_fn
        self.feats_labels_col = feats_labels_col
        self.feats_cat_replace_col = feats_cat_replace_col
        self.feats_cat_colors = feats_cat_colors
        self.feats_cat_encoding = feats_cat_encoding
        self.feats_cat_keep = feats_cat_keep
        self.feats_cat_embed_dim = feats_cat_embed_dim

        self.target = target
        if target_label is None:
            self.target_label = target
        else:
            self.target_label = target_label
        self.target_classes_fn = target_classes_fn

        self.duration = duration
        if duration_label is None:
            self.duration_label = duration
        else:
            self.duration_label = duration_label

        self.data_fn = data_fn
        self.data_index = data_index
        self.data_imputation = data_imputation

        self.split_by = split_by
        self.split_trn_val = split_trn_val
        self.split_top_feat = split_top_feat
        self.split_explicit_feat = split_explicit_feat

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.weighted_sampler = weighted_sampler

        self.dataloaders_evaluate = False

        self.dataset_trn: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_tst: Optional[Dataset] = None

        file_ext = pathlib.Path(self.data_fn).suffix
        if file_ext == '.xlsx':
            self.data_all = pd.read_excel(f"{self.data_fn}")
        elif file_ext == ".pkl":
            self.data_all = pd.read_pickle(f"{self.data_fn}")

        df_feats_con = pd.read_excel(self.feats_con_fn, index_col=0)
        self.feats_con = df_feats_con.index.values.tolist()
        self.feats_labels = {}
        for f in  self.feats_con:
            if self.feats_labels_col in df_feats_con.columns:
                self.feats_labels[f] = df_feats_con.at[f, self.feats_labels_col]
            else:
                self.feats_labels[f] = f

        if self.feats_cat_fn is not None:
            df_feats_cat = pd.read_excel(self.feats_cat_fn, index_col=0)
            self.feats_cat = df_feats_cat.index.values.tolist()
        else:
            self.feats_cat = []
            df_feats_cat = None

        if self.task in ['classification', 'survival']:
            self.target_classes_df = pd.read_excel(self.target_classes_fn)
            self.target_classes_dict = {}
            for cl_id, cl in enumerate(self.target_classes_df.loc[:, self.target].values):
                self.target_classes_dict[cl] = cl_id
            self.data_all = self.data_all.loc[self.data_all[self.target].isin(self.target_classes_dict), :]
            self.data_all[f'{self.target}_origin'] = self.data_all[self.target]
            self.data_all[self.target].replace(self.target_classes_dict, inplace=True)
            self.target_classes_num = self.data_all[self.target].nunique()
            self.data_all.reset_index(drop=True, inplace=True)
        elif self.task == 'regression':
            self.data_all = self.data_all.astype({self.target: 'float32'})

        if self.task == 'survival':
            self.data_all = self.data_all.astype({self.duration: 'float32', self.target: 'int32'})

        self.widedeep = {'cat_embed_input': None}
        if len(self.feats_cat) > 0:
            self.dict_cat_replace = {}
            self.dict_colors = {}
            if self.feats_cat_encoding == "label": # pytorch doesn't work with strings
                self.widedeep['cat_embed_input'] = []
                for f in self.feats_cat:
                    if self.feats_labels_col in df_feats_cat.columns and self.feats_cat_replace_col in df_feats_cat.columns:
                        self.dict_cat_replace[f] = ast.literal_eval(df_feats_cat.at[f, self.feats_cat_replace_col])
                        self.data_all[f].replace(self.dict_cat_replace[f], inplace=True)
                        dict_change = dict(zip(self.data_all[f].astype('category').cat.codes, self.data_all[f]))
                        self.dict_cat_replace[f] = dict_change
                    self.data_all[f] = self.data_all[f].astype('category').cat.codes.astype('int32')
                    self.widedeep['cat_embed_input'].append((f, self.data_all[f].value_counts().shape[0], self.feats_cat_embed_dim))
                    if self.feats_cat_colors in df_feats_cat.columns:
                        self.dict_colors[f] = ast.literal_eval(df_feats_cat.at[f, self.feats_cat_colors])
                    if self.feats_labels_col in df_feats_cat.columns:
                        self.feats_labels[f] = df_feats_cat.at[f, self.feats_labels_col]
                    else:
                        self.feats_labels[f] = f

                if not self.feats_cat_keep:
                    self.feats_con += self.feats_cat
                    self.feats_cat = []

            elif self.feats_cat_encoding in ["one_hot", "one_hot_drop_first"]:
                feats_cat_new = []
                for f in self.feats_cat:
                    if self.feats_cat_encoding == "one_hot_drop_first":
                        one_hot = pd.get_dummies(self.data_all.loc[:, [f]], columns=[f], drop_first=True)
                    else:
                        one_hot = pd.get_dummies(self.data_all.loc[:, [f]], columns=[f])
                    self.data_all = self.data_all.join(one_hot)
                    feats_encoded = one_hot.columns.values.tolist()

                    if self.feats_cat_keep:
                        feats_cat_new += feats_encoded
                    else:
                        self.feats_con += feats_encoded

                    if self.feats_labels_col in df_feats_cat.columns and self.feats_cat_replace_col in df_feats_cat.columns:
                        dict_cat_replace = ast.literal_eval(df_feats_cat.at[f, self.feats_cat_replace_col])
                        for f_e in feats_encoded:
                            right_part = ast.literal_eval(f_e.replace(f"{f}_", ''))
                            self.feats_labels[f_e] = f"{df_feats_cat.at[f, self.feats_labels_col]} ({dict_cat_replace[right_part]})"
                            self.dict_cat_replace[f_e] = {1: 'Yes', 0: 'No'}
                    else:
                        for f_e in feats_encoded:
                            self.feats_labels[f_e] = f_e

                if self.feats_cat_keep:
                    self.feats_cat = feats_cat_new
                else:
                    self.feats_cat = []

            else:
                raise ValueError(f"Unsupported cat_encoding: {self.feats_cat_encoding}")
        self.feats_all = self.feats_con + self.feats_cat
        self.feats_all_ids = np.arange(len(self.feats_all))
        self.feats_con_ids = np.arange(len(self.feats_con))
        self.feats_cat_ids = np.arange(len(self.feats_cat)) + len(self.feats_con)
        self.widedeep['column_idx'] = {feat_name: feat_id for feat_id, feat_name in enumerate(self.feats_all)}
        self.widedeep['continuous_cols'] = self.feats_con

        is_nans = self.data_all.loc[:, self.feats_con].isnull().values.any()
        if is_nans:
            n_nans = self.data_all.loc[:, self.feats_con].isna().sum().sum()
            log.info(f"Perform imputation for {n_nans} missed values")
            self.data_all.loc[:, self.feats_con] = self.data_all.loc[:, self.feats_con].astype('float')
            if self.data_imputation == "median":
                imputed = imp.median(self.data_all.loc[:, self.feats_con].values)
            elif self.data_imputation == "mean":
                imputed = imp.mean(self.data_all.loc[:, self.feats_con].values)
            elif self.data_imputation == "fast_knn":
                imputed = imp.fast_knn(self.data_all.loc[:, self.feats_con].values)
            elif self.data_imputation == "random":
                imputed = imp.random(self.data_all.loc[:, self.feats_con].values)
            elif self.data_imputation == "mice":
                imputed = imp.mice(self.data_all.loc[:, self.feats_con].values)
            elif self.data_imputation == "em":
                imputed = imp.em(self.data_all.loc[:, self.feats_con].values)
            elif self.data_imputation == "mode":
                imputed = imp.mode(self.data_all.loc[:, self.feats_con].values)
            else:
                raise ValueError(f"Unsupported imputation: {self.data_imputation}")
            self.data_all.loc[:, self.feats_con] = imputed
        self.data_all.loc[:, self.feats_con] = self.data_all.loc[:, self.feats_con].astype('float32')

        self.data_all['ids'] = self.data_all.index.values
        self.ids_trn_val = self.data_all.index[self.data_all[self.split_explicit_feat].isin(["trn_val", "trn", "val"])].values

        self.ids_tst = {}
        tst_set_names = [x for x in self.data_all[self.split_explicit_feat].unique() if x.startswith("tst")]
        for tst_set_name in tst_set_names:
            self.ids_tst[tst_set_name] = self.data_all.index[self.data_all[self.split_explicit_feat] == tst_set_name].values
        if len(tst_set_names) > 1:
            self.ids_tst['tst_all'] = self.data_all.index[self.data_all[self.split_explicit_feat].str.startswith("tst")].values

        self.colors = {'trn': px.colors.qualitative.Dark24[0], 'val': px.colors.qualitative.Dark24[1]}
        for tst_set_name_index, tst_set_name in enumerate(self.ids_tst):
            self.colors[tst_set_name] = px.colors.qualitative.Dark24[tst_set_name_index + 2]

        self.data_all.set_index(self.data_index, inplace=True)

        self.dataset = TabularDataset(
            data=self.data_all,
            feats_con=self.feats_con,
            feats_cat=self.feats_cat,
            target=self.target
        )

        if self.task == 'survival':
            self.data_all[self.feats_cat] = self.data_all[self.feats_cat].astype('float32')
        else:
            self.data_all[self.feats_cat] = self.data_all[self.feats_cat].astype('int32')

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def refresh_datasets(self):
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        log.info(f"trn count: {len(self.dataset_trn)}")
        self.dataset_val = Subset(self.dataset, self.ids_val)
        log.info(f"val count: {len(self.dataset_val)}")
        self.dataset_tst = {}
        for tst_set_name in self.ids_tst:
            self.dataset_tst[tst_set_name] = Subset(self.dataset, self.ids_tst[tst_set_name])
            log.info(f"{tst_set_name} count: {len(self.dataset_tst[tst_set_name])}")

    def perform_split(self):
        if self.split_by == "explicit_feat":
            self.ids_trn = self.data_all.loc[self.data_all[self.split_explicit_feat] == "trn", 'ids'].values
            self.ids_val = self.data_all.loc[self.data_all[self.split_explicit_feat] == "val", 'ids'].values
        else:
            assert abs(1.0 - sum(self.split_trn_val)) < 1.0e-8, "Sum of trn_val_split must be 1"
            target_trn_val = self.data_all.loc[self.data_all.index[self.ids_trn_val], self.target].values
            if self.task in ['classification', 'survival']:
                self.ids_trn, self.ids_val = train_test_split(
                    self.ids_trn_val,
                    test_size=self.split_trn_val[1],
                    stratify=target_trn_val,
                    random_state=self.seed
                )
            elif self.task == 'regression':
                ptp = np.ptp(target_trn_val)
                num_bins = 4
                bins = np.linspace(
                    np.min(target_trn_val) - 0.1 * ptp,
                    np.max(target_trn_val) + 0.1 * ptp,
                    num_bins + 1
                )
                binned = np.digitize(target_trn_val, bins) - 1
                unique, counts = np.unique(binned, return_counts=True)
                occ = dict(zip(unique, counts))
                log.info(f"regression stratification: {occ}")
                self.ids_trn, self.ids_val = train_test_split(
                    self.ids_trn_val,
                    test_size=self.split_trn_val[1],
                    stratify=binned,
                    random_state=self.seed
                )

        self.refresh_datasets()

    def plot_split_classification(self, suffix: str = ''):

        df_fig = self.data_all.loc[:, [self.target, self.split_explicit_feat]].copy()
        df_fig.rename(columns={self.target: self.target_label}, inplace=True)
        target_dict = {v: k for k, v in self.target_classes_dict.items()}
        df_fig.replace({self.target_label: target_dict}, inplace=True)
        df_fig.loc[df_fig.index[self.ids_trn], "Part"] = 'trn'
        df_fig.loc[df_fig.index[self.ids_val], "Part"] = 'val'
        order = ['trn', 'val']
        for tst_set_name in self.ids_tst:
            if tst_set_name != 'tst_all':
                df_fig.loc[df_fig.index[self.ids_tst[tst_set_name]], "Part"] = tst_set_name
                order.append(tst_set_name)

        plt.figure()
        sns.set_theme(style='whitegrid', font_scale=1)
        bar = sns.countplot(
            data=df_fig,
            x=f"Part",
            order=order,
            hue=self.target_label,
            palette=px.colors.qualitative.Dark24,
            edgecolor='black'
        )
        for x in bar.containers:
            bar.bar_label(x)
        bar.set_ylabel("Count")
        plt.savefig(f"count.png", bbox_inches='tight')
        plt.savefig(f"count.pdf", bbox_inches='tight')
        plt.close()

    def plot_split_regression(self, target, target_label, suffix: str = ''):

        df_fig = self.data_all.loc[:, [target, self.split_explicit_feat]].copy()
        df_fig.rename(columns={target: target_label}, inplace=True)
        df_fig.loc[df_fig.index[self.ids_trn], "Part"] = 'trn'
        df_fig.loc[df_fig.index[self.ids_val], "Part"] = 'val'
        for tst_set_name in self.ids_tst:
            if tst_set_name != 'tst_all':
                df_fig.loc[df_fig.index[self.ids_tst[tst_set_name]], "Part"] = tst_set_name

        hist_bins = np.linspace(0, 120, 13)

        hue_order = ['trn', 'val'] + [x for x in self.ids_tst.keys() if x != 'tst_all']

        fig = plt.figure()
        sns.set_theme(style='whitegrid')
        sns.histplot(
            data=df_fig,
            bins=hist_bins,
            edgecolor='k',
            linewidth=1,
            x=target_label,
            hue="Part",
            hue_order=hue_order[::-1],
            palette=self.colors,
            multiple="stack",
        )
        plt.savefig(f"hist{suffix}.png", bbox_inches='tight', dpi=400)
        plt.savefig(f"hist{suffix}.pdf", bbox_inches='tight')
        plt.close(fig)

    def plot_split(self, suffix: str = ''):

        if self.task == 'classification':
            self.plot_split_classification(suffix=suffix)
        elif self.task == 'regression':
            self.plot_split_regression(suffix=suffix, target=self.target, target_label=self.target_label)
        elif self.task == 'survival':
            self.plot_split_classification(suffix=suffix)
            self.plot_split_regression(suffix=suffix, target=self.duration, target_label=self.duration_label)

    def get_cross_validation_df(self):
        columns = ['ids', self.target]
        if self.split_top_feat is not None:
            columns.append(self.split_top_feat)
        cross_validation_df = self.data_all.loc[self.data_all.index[self.ids_trn_val], columns]
        return cross_validation_df

    def train_dataloader(self):
        if self.dataloaders_evaluate:
            return DataLoader(
                dataset=self.dataset_trn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False
            )
        else:
            ys_trn = self.dataset.y[self.ids_trn]
            if self.task in ['classification', 'survival'] and self.weighted_sampler:
                class_counter = Counter(ys_trn)
                class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
                weights = torch.FloatTensor([class_weights[y] for y in ys_trn])
                weighted_sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True
                )
                return DataLoader(
                    dataset=self.dataset_trn,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    sampler=weighted_sampler
                )
            else:
                return DataLoader(
                    dataset=self.dataset_trn,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=True
                )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloaders(self):
        dataloaders = {}
        for tst_set_name in self.dataset_tst:
            dataloaders[tst_set_name] = DataLoader(
                dataset=self.dataset_tst[tst_set_name],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        return dataloaders

    def get_features(self):
        features = {
            'all': self.feats_all,
            'con': self.feats_con,
            'cat': self.feats_cat,
            'all_ids': self.feats_all_ids,
            'con_ids': self.feats_con_ids,
            'cat_ids': self.feats_cat_ids,
            'labels': self.feats_labels
        }
        return features

    def get_widedeep(self):
        return self.widedeep

    def get_class_names(self):
        return list(self.target_classes_dict.keys())

    def get_data(self):
        return self.data_all
