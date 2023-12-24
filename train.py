from subprocess import check_output

from dvc.api import DVCFileSystem
from hydra import main
from joblib import dump
from mlflow import log_metric, log_param, set_tracking_uri, start_run
from numpy import mean
from omegaconf import DictConfig
from pandas import concat, get_dummies, read_csv
from sklearn.metrics import mean_absolute_percentage_error
from toml import load
from torch import from_numpy, nn, optim, utils
from torcheval.metrics import R2Score


class SetUpData(utils.data.Dataset):
    def __init__(self, path_file):
        self.df = read_csv(path_file, index_col=None)

        dummy_fields = ["Pclass", "Sex", "Embarked"]
        for field in dummy_fields:
            dummies = get_dummies(self.df[field], prefix=field, drop_first=False)
            self.df = concat([self.df, dummies], axis=1)

        fields_to_drop = [
            "PassengerId",
            "Cabin",
            "Pclass",
            "Name",
            "Sex",
            "Ticket",
            "Embarked",
        ]
        self.df = self.df.drop(fields_to_drop, axis=1)

        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].mean())

        to_normalize = ["Age", "Fare"]
        for field in to_normalize:
            mean_value, std = self.df[field].mean(), self.df[field].std()
            self.df.loc[:, field] = (self.df[field] - mean_value) / std

        self.X = from_numpy(self.df.values[:, 1:13]).float()
        self.y = from_numpy(self.df.values[:, 0]).float()

        self.n_samples = self.df.shape[0]

    def __len__(self):  # Length of the dataset.
        return self.X.size()[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


@main(version_base=None, config_path="conf", config_name="config")
def train_model(cfg: DictConfig):
    config_path = "mlflow_config.toml"
    config = load(config_path)
    set_tracking_uri(f"http://{config['server']['host']}:{config['server']['port']}")
    git_commit_id = (
        check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    )
    with start_run():
        log_param("git_commit_id", git_commit_id)
        log_param("batch_size", cfg["params"].batch_size)
        log_param("learning_rate", cfg["params"].learning_rate)
        log_param("epochs", cfg["params"].epochs)
        fs = DVCFileSystem("https://github.com/blarno/mlops_project1/")
        with fs.open("data/titanic.csv") as f:
            data = SetUpData(f)
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size

        train, test = utils.data.random_split(data, [train_size, test_size])
        data_train = utils.data.DataLoader(train, batch_size=cfg["params"].batch_size)
        dump(data.X[test.indices], "data/X_test.h5")

        model = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg["params"].learning_rate)
        for i in range(cfg["params"].epochs):
            losses = []
            mape = []
            r2score = []
            for X, y in data_train:
                pred = model(X).view(-1)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())
                mape.append(mean_absolute_percentage_error(pred.detach().numpy(), y))
                r2score.append(R2Score().update(pred, y).compute())
            log_metric("MAPE_train", mean(mape), step=i)
            log_metric("R2Score_train", mean(r2score), step=i)
            log_metric("MSELoss_train", mean(losses), step=i)
        dump(model, "models/model.h5")


if __name__ == "__main__":
    train_model()
