import pandas as pd
import lightning as L
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from src.datasets.ts_dataset import TSDataset


class TSDataModule(L.LightningDataModule):
    def __init__(
        self,
        task_name,
        data_folder,
        filename,
        selected_cols,
        batch_size,
        num_workers,
        pred_idx,
        dataset,
        data_index=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.test_dataset = None
        self.scaler = StandardScaler()
        self.data_index = data_index

    def setup(self, stage):
        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        filepath = f"{self.hparams.data_folder}/{self.hparams.task_name}_{self.hparams.filename}.csv"
        print(f"[filepath] {filepath}")
        df = pd.read_csv(filepath)
        dt = pd.to_datetime(df["dt"])
        df = df.drop(columns=["dt"])

        n = len(df)
        if self.data_index is not None:
            train_end = self.data_index[0]
            val_end = self.data_index[1]
            test_end = self.data_index[2]
        else:
            train_end = n - 24 * 30
            val_end = n - 24 * 10
            test_end = n
            # train_end = int(n * 0.8)
            # val_end = n - int(n * 0.1)
            # test_end = n

        train_df = df[:train_end]
        self.scaler.fit(df)
        df = scale_df(df, self.scaler)
        print(f"{df.head()}")

        split_index = dict(
            train=[0, train_end],
            val=[train_end, val_end],
            test=[val_end, test_end],
            pred=[self.hparams.pred_idx, val_end],
        )

        if stage == "fit" or stage is None:
            self.train_dataset = TSDataset(
                split_index, dt, df, mode="train", **self.hparams.dataset
            )
            self.val_dataset = TSDataset(
                split_index, dt, df, mode="val", **self.hparams.dataset
            )
            self.pred_dataset = TSDataset(
                split_index, dt, df, mode="pred", **self.hparams.dataset
            )
        if stage == "test" or stage is None:
            self.test_dataset = TSDataset(
                split_index, dt, df, mode="test", **self.hparams.dataset
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            # num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            # num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            # num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset, batch_size=self.hparams.batch_size, shuffle=False
        )

    # @property
    # def num_features(self):
    #     return self.train_dataset.num_features

    # @property
    # def feature_names(self):
    #     return self.train_dataset.feature_names
