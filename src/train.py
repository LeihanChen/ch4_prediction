from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TimeSeriesTransformerForPrediction,
    TimeSeriesTransformerConfig,
)
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import hydra
import os
from omegaconf import DictConfig, OmegaConf
import wandb
wandb.login()

class TimeSeriesCh4Dataset(Dataset):
    def __init__(self, data, past_length, future_length):
        self.ch4_data = data
        self.past_length = past_length
        self.future_length = future_length

    def __len__(self):
        return len(self.ch4_data) - self.past_length - self.future_length

    def __getitem__(self, idx):
        # Implement how to load and process a single data point here
        item_past = self.ch4_data.loc[idx : idx + self.past_length,
            ["CH4", "CH4_true", "T", "P", "RH"]
        ]
        past_values = torch.tensor(item_past["CH4"].values, dtype=torch.float32)
        past_time_features = torch.tensor(item_past[["T", "P", "RH"]].values, dtype=torch.float32)
        future_values = torch.tensor(self.ch4_data.loc[
            idx + self.past_length : idx + self.future_length + self.past_length
        , ["CH4_true"]].values, dtype=torch.float32)
        future_time_features = torch.tensor(self.ch4_data.loc[
            idx + self.past_length : idx + self.future_length + self.past_length
        , ["T", "P", "RH"]].values, dtype=torch.float32)

        # Return a dictionary with features
        return {
            "past_values": past_values,
            "past_time_features": past_time_features,
            "future_values": future_values,
            "future_time_features": future_time_features,
        }


def create_dataset(file_path: str):
    column_data_types = {
    'date': 'str',     # For string values
    'time(UTC)': 'str',   # For string values
    'CH4': 'float',     # For floating-point values
    'CH4_true': 'float',     # For floating-point values
    'T': 'float',     # For floating-point values
    'P': 'float',    # For floating-point values
    'RH': 'float',   # For floating-point values
    }
    df_data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path),
        names=["date", "time(UTC)", "CH4", "CH4_true", "T", "P", "RH"],
        sep=" ",
        header=0,
        dtype=column_data_types,
        # parse_dates=["date", "time(UTC)"],
        # infer_datetime_format=True,
        error_bad_lines=False,
        na_values=["nan", "5G4.77"],
    )
    df_data = df_data.dropna(how="any")

    df_data["utc_time"] = df_data["date"] + df_data["time(UTC)"]
    df_data.drop("date", inplace=True, axis=1)
    df_data.drop("time(UTC)", inplace=True, axis=1)

    # Split the DataFrame into training (60%), validation (20%), and test (20%) sets
    train_df, temp_df = train_test_split(df_data, test_size=0.4, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Fit the scaler to your data and transform the data
    scaler = StandardScaler()
    train_fit = scaler.fit(train_df[["T", "P", "RH"]])
    train_df[["T", "P", "RH"]] = train_fit.transform(train_df[["T", "P", "RH"]])
    valid_df[["T", "P", "RH"]] = train_fit.transform(valid_df[["T", "P", "RH"]])
    test_df[["T", "P", "RH"]] = train_fit.transform(test_df[["T", "P", "RH"]])

    # Create HF dataset from pd dataframe
    # train_dataset = Dataset.from_pandas(train_df).shuffle(seed=42)
    # valid_dataset = Dataset.from_pandas(valid_df).shuffle(seed=42)
    # test_dataset = Dataset.from_pandas(test_df).shuffle(seed=42)

    # Combine them into a single DatasetDict
    # data_dict = DatasetDict(
    #     {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
    # )

    # data_dict["train"] = data_dict["train"].set_format(
    #     type="torch", columns=["CH4", "CH4_true", "T", "P", "RH"]
    # )
    # data_dict["validation"] = data_dict["validation"].set_format(
    #     type="torch", columns=["CH4", "CH4_true", "T", "P", "RH"]
    # )
    # data_dict["test"] = data_dict["test"].set_format(
    #     type="torch", columns=["CH4", "CH4_true", "T", "P", "RH"]
    # )
    return train_df, valid_df, test_df


@hydra.main(config_path="../config", config_name="config.yaml")
def train_ch4_time_series_transformer(cfg: DictConfig):
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=OmegaConf.to_container(cfg.model, resolve=True))
    print(OmegaConf.to_yaml(cfg))
    
    # Randomly initializing a model (with random weights) from the configuration
    model = TimeSeriesTransformerForPrediction(set_time_seres_tf_model_config(cfg))
    # Get the configuration of model
    # configuration = model.config
    
    # TODO(Leihan): Moving training into pytorch lightning
    # Train the model
    train_df, _, _ = create_dataset(cfg.data.file_path)
    train(model, train_df, cfg)

    # Prediction on the test dataset.

def set_time_seres_tf_model_config(config: DictConfig):
    configuration = TimeSeriesTransformerConfig(
        prediction_length=config.model.prediction_length,
        context_length=config.model.context_length,
        input_size=config.model.input_size,
        scaling=config.model.scaling,
        num_dynamic_real_features=config.model.num_dynamic_real_features,
        num_parallel_samples=config.model.num_parallel_samples,
        num_time_features=0,
    )

    return configuration


def train(model, dataset, config: DictConfig):
    ch4_dataset = TimeSeriesCh4Dataset(
        dataset, config.model.past_length, config.model.prediction_length
    )
    dataloader = torch.utils.data.DataLoader(ch4_dataset, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    # criterion = nn.MSELoss()

    num_epoches = config.training.epochs

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
    else:
        device = torch.device("cpu")  # Use the CPU
    model.to(device)

    wandb.watch(model, log_freq=50)
    model.train()

    for _ in range(num_epoches):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                past_values=batch["past_values"].to(device),
                past_time_features=batch["past_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    train_ch4_time_series_transformer()
    
