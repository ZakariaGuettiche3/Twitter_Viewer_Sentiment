import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import json
import yaml
import logging
import mlflow
from mlflow.models import infer_signature
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('log/errors_model_evaluation.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_param(param_path : str) -> dict:
    try:
       with open( param_path, "r") as f :
            params = yaml.safe_load(f)
       logger.debug('Parameters retrieved from %s', param_path)
       return params
    except FileNotFoundError:
       logger.error("File note found %s" , param_path)
       raise
    except Exception as e:
       logger.error("Unexpected error while loading %s" , e)
       raise
         
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        df = df.dropna()
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    
def encode_sentence(sentence , MAX_LEN , word2idx):
    
    ids = []

    for word in sentence.split():

        ids.append(
            word2idx.get(word,1)
        )

    ids = ids[:MAX_LEN]

    ids += [0]*(MAX_LEN-len(ids))

    return ids

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        encoder = LabelEncoder()
        df["category"] = encoder.fit_transform(df["category"])
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while encoding the data: %s', e)
        raise


class TextDataset(Dataset):
    
    def __init__(self, texts, labels,max_len,word2idx):

        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.max_len = max_len
        self.word2idx = word2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        sentence = encode_sentence(self.texts[idx],self.max_len,self.word2idx)

        return (
            torch.tensor(sentence, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
def load_test_data(X_test ,y_test , BATCH_SIZE, suf,max_len ,word2idx) :
    try:
       train_dataset = TextDataset(
             X_test,
             y_test,
             max_len,
             word2idx
             
        )

       test_loader = DataLoader(
             train_dataset,
             batch_size=BATCH_SIZE,
             shuffle=suf,
             num_workers=0,
             pin_memory=True
            )


       logger.debug('Parameters retrieved from %s')
       return train_dataset , test_loader
    except Exception as e:
       logger.error("Unexpected error while loading %s" , e)
       raise
   
   
class BiGRUClassifier(nn.Module):
    
    def __init__(
        self,
        embedding_matrix,
        hidden_size,
        hidden_size_2,
        num_classes,
        dropout=0.5,
        freeze_embedding=True
    ):

        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embedding,
            padding_idx=0
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_64 = nn.LayerNorm(hidden_size)
        self.norm_32 = nn.LayerNorm(hidden_size_2)

        self.fc1 = nn.Linear(
            hidden_size,
            hidden_size_2
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            hidden_size_2,
            num_classes
        )

    def forward(self, x):

        x = self.embedding(x)
    
        _, hidden = self.gru(x)
        
        hidden = hidden[-1]       
        
        hidden = self.norm_64(hidden)

        hidden = self.dropout(hidden)

        hidden = self.fc1(hidden)

        hidden = self.norm_32(hidden)

        hidden = self.relu(hidden)

        hidden = self.dropout(hidden)

        logits = self.fc2(hidden)

        return logits

def load_model(
    model_path: str,
    embedding_matrix,
    hidden_size: int,
    hidden_size_2: int,
    dropout: float,
    num_classes: int,
    device: torch.device,
):
  
    try:
        logger.info("Loading model from %s", model_path)

        model = BiGRUClassifier(
            embedding_matrix=embedding_matrix,
            hidden_size=hidden_size,
            hidden_size_2=hidden_size_2,
            dropout=dropout,
            num_classes=num_classes,
        )

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            )
        )

        model.to(device)
        model.eval()

        logger.info("Model loaded successfully")

        return model

    except FileNotFoundError:
        logger.exception("Model file not found: %s", model_path)
        raise

    except Exception:
        logger.exception("Failed to load model")
        raise
    
def test_model(model, dataloader, device):
 

    try:
        logger.info("Starting model evaluation")

        model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():

            progress_bar = tqdm(
                dataloader,
                desc="Testing",
                leave=False,
            )

            for batch_idx, (inputs, labels) in enumerate(progress_bar, start=1):

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)

                probabilities = torch.softmax(outputs, dim=1)

                predictions = outputs.argmax(dim=1)

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

                all_probabilities.extend(
                    probabilities.cpu().numpy()
                )

                if batch_idx % 100 == 0:
                    logger.debug(
                        "Processed batch %d/%d",
                        batch_idx,
                        len(dataloader),
                    )

        logger.info(
            "Evaluation completed successfully. Processed %d samples.",
            len(all_labels),
        )

        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities),
        )

    except Exception:
        logger.exception("Error during model evaluation.")
        raise
    
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise
    



def evaluate_model(
    y_true,
    y_pred,
    y_prob,
    output_dir: str = "metrics",
):

    try:
        logger.info("Starting model evaluation...")

        os.makedirs(output_dir, exist_ok=True)

        # ---------------- Metrics ---------------- #

        metrics = {
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="weighted",
                )
            ),
        }

        logger.info(
            "Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )


        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(7, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, "confusion_matrix.png"),
            dpi=300,
        )

        plt.close()

        logger.info("Confusion matrix saved.")

        logger.info("Evaluation completed successfully.")

        return metrics

    except Exception:
        logger.exception("Failed to evaluate the model.")
        raise
    
def main():
    try:
        logger.info("Starting model evaluation pipeline...")

        mlflow.set_tracking_uri(
            "http://ec2-13-50-105-122.eu-north-1.compute.amazonaws.com:5000/"
        )

        mlflow.set_experiment("dvc_pipline_run")
        
        path = mlflow.set_experiment("dvc_pipline_run")

        with mlflow.start_run() as run:

            params = load_param("params.yaml")

            hidden_dim = params["model"]["hidden_dim"]
            hidden_dim_2 = params["model"]["hidden_dim_2"]
            dropout = params["model"]["dropout"]
            max_len = params["word2vec"]["max_len"]

            batch_size = params["training"]["batch_size"]

            logger.info("Loading test dataset...")

            test_df = load_data(
                "Data/preprocessed/test_preprocessed.csv"
            )
            
            test_df = encode_data(test_df)

            logger.info("Loading Word2Vec artifacts...")

            with open(
                "word2vec/word2idx.json",
                "r",
                encoding="utf-8",
            ) as f:
                word2idx = json.load(f)

            embedding_matrix = np.load(
                "word2vec/embedding_matrix.npy"
            )

            num_classes = len(
                np.unique(test_df["category"])
            )

            _, test_loader = load_test_data(
                X_test=test_df["clean_text"],
                y_test=test_df["category"],
                BATCH_SIZE=batch_size,
                suf=False,
                max_len=max_len,
                word2idx=word2idx,
            )

            model = load_model(
                model_path="best_bigru_model.pth",
                embedding_matrix=embedding_matrix,
                hidden_size=hidden_dim,
                hidden_size_2=hidden_dim_2,
                dropout=dropout,
                num_classes=num_classes,
                device=device,
            )

            predictions, labels, probabilities = test_model(
                model,
                test_loader,
                device,
            )

            metrics = evaluate_model(
                y_true=labels,
                y_pred=predictions,
                y_prob=probabilities,
                output_dir="metrics",
            )

            mlflow.log_params({
                "hidden_dim": hidden_dim,
                "hidden_dim_2": hidden_dim_2,
                "dropout": dropout,
                "batch_size": batch_size,
            })

            mlflow.log_metrics(metrics)

            mlflow.log_artifact(
                "metrics/confusion_matrix.png"
            )

            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                serialization_format="pickle" ,
                signature=infer_signature(
                    np.zeros((1, max_len), dtype=np.int64),
                    probabilities,
                      
                ),
            )

            save_model_info(
                run_id=run.info.run_id,
                model_path=model_info.model_uri,
                file_path="models/model_info.json",
            )

            logger.info("Model evaluation completed successfully.")

    except Exception:
        logger.exception("Evaluation pipeline failed.")
        raise


if __name__ == "__main__":
    main()
        