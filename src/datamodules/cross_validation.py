from abc import abstractmethod, ABC
from sklearn.model_selection import RepeatedStratifiedKFold
from src.datamodules.tabular import TabularDataModule
import numpy as np
from src.utils import utils
from itertools import combinations
from math import comb


log = utils.get_logger(__name__)

class CVSplitter(ABC):

    def __init__(
            self,
            datamodule: TabularDataModule,
            is_split: bool = True,
            n_splits: int = 5,
            n_repeats: int = 5,
            random_state: int = 42,
            all_combinations: bool = False,
    ):
        self.datamodule = datamodule
        self.is_split = is_split
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.all_combinations = all_combinations

    @abstractmethod
    def split(self):
        pass


class RepeatedStratifiedKFoldCVSplitter(CVSplitter):

    def __init__(
            self,
            datamodule: TabularDataModule,
            is_split: bool = True,
            n_splits: int = 5,
            n_repeats: int = 5,
            random_state: int = 42,
            all_combinations: bool = False,
    ):
        super().__init__(datamodule, is_split, n_splits, n_repeats, random_state, all_combinations)
        self.k_fold = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

    def split(self):

        if self.datamodule.split_by == "explicit_feat":
            self.is_split = False

        if self.is_split:
            self.datamodule.setup()
            cross_validation_df = self.datamodule.get_cross_validation_df()
            ids = cross_validation_df.loc[:, 'ids'].values
            target = cross_validation_df.loc[:, self.datamodule.target].values
            if self.datamodule.split_by != "top_feat":
                if self.datamodule.task in ['classification', 'survival']:
                    splits = self.k_fold.split(X=ids, y=target, groups=target)
                elif self.datamodule.task == "regression":
                    ptp = np.ptp(target)
                    num_bins = 4
                    bins = np.linspace(np.min(target) - 0.1 * ptp, np.max(target) + 0.1 * ptp, num_bins + 1)
                    binned = np.digitize(target, bins) - 1
                    unique, counts = np.unique(binned, return_counts=True)
                    occ = dict(zip(unique, counts))
                    log.info(f"Regression stratification: {occ}")
                    splits = self.k_fold.split(X=ids, y=binned, groups=binned)
                else:
                    raise ValueError(f'Unsupported self.datamodule.task: {self.datamodule.task}')

                for ids_trn, ids_val in splits:
                    yield ids[ids_trn], ids[ids_val]

            else:
                top_feat_vals = cross_validation_df[self.datamodule.split_top_feat].unique()
                if self.datamodule.task in ['classification', 'survival']:
                    if self.all_combinations:
                        n_tst = len(top_feat_vals) // self.n_splits
                        log.info(f"All combinations params: {n_tst} from {len(top_feat_vals)}. Total {comb(len(top_feat_vals), n_tst)} combinations")
                        all_vals = list(range(len(top_feat_vals)))
                        tst_vals = combinations(all_vals, n_tst)
                        splits = []
                        for tst_val in tst_vals:
                            trn_val = list(set(all_vals) - set(tst_val))
                            splits.append((trn_val, list(tst_val)))
                    else:
                        splits = self.k_fold.split(X=top_feat_vals, y=np.ones(len(top_feat_vals)))
                elif self.datamodule.task == "regression":
                    raise ValueError(f'Unsupported split by feature for the regression')
                else:
                    raise ValueError(f'Unsupported self.datamodule.task: {self.datamodule.task}')

                for ids_trn_feat, ids_val_feat in splits:
                    trn_values = top_feat_vals[ids_trn_feat]
                    ids_trn = cross_validation_df.loc[cross_validation_df[self.datamodule.split_top_feat].isin(trn_values), 'ids'].values
                    val_values = top_feat_vals[ids_val_feat]
                    ids_val = cross_validation_df.loc[cross_validation_df[self.datamodule.split_top_feat].isin(val_values), 'ids'].values
                    yield ids_trn, ids_val

        else:
            yield self.datamodule.ids_trn, self.datamodule.ids_val
