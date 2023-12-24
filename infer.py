from dvc.api import DVCFileSystem
from fire import Fire
from joblib import load
from pandas import DataFrame


def predict():
    fs = DVCFileSystem("https://github.com/blarno/mlops_project1/")
    with fs.open("data/X_test.h5") as f:
        x_test = load(f)
    with fs.open("models/model.h5") as f:
        model = load(f)

    preds = DataFrame(
        model.forward(x_test).view(-1).detach().numpy(), columns=["preds"]
    )
    return preds.to_csv("data/predict.csv")


if __name__ == "__main__":
    Fire(predict)
