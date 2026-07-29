import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import json
import yaml
import logging
from tqdm.auto import tqdm
import os 
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
logger = logging.getLogger('model_build')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('log/errors.log')
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
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    
def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        encoder = LabelEncoder()
        df["category"] = encoder.fit_transform(df["category"])
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while encoding the data: %s', e)
        raise


def calculate_weights(
    df: pd.DataFrame
) -> torch.Tensor:

    try:
        logger.info("Calculating class weights...")

        classes = np.sort(df["category"].unique())

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=df["category"]
        )

        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.debug("Detected classes: %s", classes)
        logger.debug("Class weights: %s", weights.tolist())

        logger.info("Class weights calculated successfully.")

        return weights

    except KeyError:
        logger.exception("Column '%s' does not exist.", "category")
        raise

    except Exception:
        logger.exception("Failed to calculate class weights.")
        raise
    


def train_word2vec(
    df: pd.DataFrame,
    text_column: str,
    embedding_dim: int,
    window: int,
    min_count: int,
    epochs: int,
) -> tuple[np.ndarray, dict]:
    
    try:
        logger.info("Training Word2Vec model...")

        train_tokens = [
            sentence.split()
            for sentence in df[text_column]
        ]

        logger.debug("Number of sentences: %d", len(train_tokens))

        model = Word2Vec(
            sentences=train_tokens,
            vector_size=embedding_dim,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs,
        )

        logger.info("Word2Vec training completed.")

        word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        for word in model.wv.index_to_key:
            word2idx[word] = len(word2idx)

        vocab_size = len(word2idx)

        logger.info("Vocabulary size: %d", vocab_size)

        embedding_matrix = np.zeros(
            (vocab_size, embedding_dim),
            dtype=np.float32,
        )

        for word, idx in word2idx.items():
            if word in model.wv:
                embedding_matrix[idx] = model.wv[word]

        logger.info("Embedding matrix created.")

        return embedding_matrix, word2idx

    except Exception:
        logger.exception("Failed to train Word2Vec.")
        raise


def save_word2vec_artifacts(
    embedding_matrix: np.ndarray,
    word2idx: dict,
    output_dir: str = "word2vec",
) -> None:
    """
    Save Word2Vec artifacts.
    """

    try:
        logger.info("Saving Word2Vec artifacts...")

        os.makedirs(output_dir, exist_ok=True)

        np.save(
            os.path.join(output_dir, "embedding_matrix.npy"),
            embedding_matrix,
        )

        with open(
            os.path.join(output_dir, "word2idx.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                word2idx,
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.info("Word2Vec artifacts saved successfully.")

    except Exception:
        logger.exception("Failed to save Word2Vec artifacts.")
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
def load_train_data(X_train ,y_train , BATCH_SIZE, suf,max_len ,word2idx) :
    try:
       train_dataset = TextDataset(
             X_train,
             y_train,
             max_len,
             word2idx
             
        )

       train_loader = DataLoader(
             train_dataset,
             batch_size=BATCH_SIZE,
             shuffle=suf,
             num_workers=0,
             pin_memory=True
            )


       logger.debug('Parameters retrieved from %s')
       return train_dataset , train_loader
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
    
class EarlyStopping:
    
    def __init__(
        self,
        patience=3,
        path="best_model.pth"
    ):

        self.patience = patience

        self.counter = 0

        self.best_loss = float("inf")

        self.path = path

        self.stop = False

    def __call__(self, loss, model):

        if loss < self.best_loss:

            self.best_loss = loss

            self.counter = 0

            torch.save(
                model.state_dict(),
                self.path
            )

        else:

            self.counter += 1

            print(
                f"EarlyStopping {self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.stop = True
                
                
def build_model(
    embedding_matrix,
    hidden_size: int,
    hidden_size_2: int,
    num_classes: int,
    learning_rate: float,
    class_weights: torch.Tensor,
    dropout: int,

):


    try:
        logger.info("Building BiGRU model...")

        model = BiGRUClassifier(
            embedding_matrix=embedding_matrix,
            hidden_size=hidden_size,
            hidden_size_2=hidden_size_2,
            num_classes=num_classes,
            dropout = dropout,
        ).to(device)

        logger.debug("Model moved to %s.", device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=0.01,
        )

        logger.debug("Loss function initialized.")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3,
        )

        logger.debug("Optimizer initialized.")

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=torch.cuda.is_available(),
        )

        logger.info("Model and training components created successfully.")

        return model, criterion, optimizer, scaler

    except Exception:
        logger.exception("Failed to build the model.")
        raise
    
def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for inputs, labels in progress_bar:

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            outputs = model(inputs)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=loss.item()
        )

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False
        )

        for inputs, labels in progress_bar:

            inputs = inputs.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(inputs)

                loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    early_stopping,
    device,
    epochs: int,
):
    """
    Train the model for multiple epochs.

    Returns
    -------
    dict
        Dictionary containing the training history.
    """

    try:
        logger.info("Starting model training...")

        train_losses = []
        valid_losses = []

        train_accuracies = []
        valid_accuracies = []

        best_accuracy = 0.0

        for epoch in range(epochs):

            logger.info(
                "Epoch %d/%d",
                epoch + 1,
                epochs,
            )

            # -------------------- Training --------------------

            train_loss, train_acc = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )

            # -------------------- Validation --------------------

            valid_loss, valid_acc = validate(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
            )

            scheduler.step(valid_loss)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            train_accuracies.append(train_acc)
            valid_accuracies.append(valid_acc)

            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Train Loss: %.4f | Train Acc: %.4f | "
                "Valid Loss: %.4f | Valid Acc: %.4f | "
                "LR: %.6f",
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                current_lr,
            )

            if valid_acc > best_accuracy:

                best_accuracy = valid_acc

                torch.save(
                    model.state_dict(),
                    "best_model.pth",
                )

                logger.info(
                    "New best model saved (Validation Accuracy: %.4f).",
                    best_accuracy,
                )

            # -------------------- Early Stopping --------------------

            early_stopping(
                valid_loss,
                model,
            )

            if early_stopping.stop:

                logger.info(
                    "Early stopping triggered after epoch %d.",
                    epoch + 1,
                )

                break

        logger.info("Training completed successfully.")

        return {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "best_accuracy": best_accuracy,
        }

    except Exception:
        logger.exception("Training pipeline failed.")
        raise
    
    
def main():
    try:
        logger.info("Starting training pipeline...")

        # ---------------- Load parameters ---------------- #

        params = load_param("params.yaml")

        embedding_dim = params["word2vec"]["embedding_dim"]
        window = params["word2vec"]["window"]
        min_count = params["word2vec"]["min_count"]
        word2vec_epochs = params["word2vec"]["epochs"]
        max_len = params["word2vec"]["max_len"]

        hidden_dim = params["model"]["hidden_dim"]
        hidden_dim_2 = params["model"]["hidden_dim_2"]
        dropout = params["model"]["dropout"]
        

        epochs = params["training"]["epochs"]
        batch_size = params["training"]["batch_size"]
        learning_rate = params["training"]["learning_rate"]

        # ---------------- Create folders ---------------- #

        os.makedirs("models", exist_ok=True)

        WORD2VEC_DIR = "word2vec"

        # ---------------- Load datasets ---------------- #

        train_df = load_data("Data/preprocessed/train_preprocessed.csv")
        valid_df = load_data("Data/preprocessed/valid_preprocessed.csv")

        # ---------------- Encode labels ---------------- #

        train_df = encode_data(train_df)
        valid_df = encode_data(valid_df)

        # ---------------- Class weights ---------------- #

        class_weights = calculate_weights(train_df)

        # ---------------- Word2Vec ---------------- #

        embedding_file = os.path.join(
            WORD2VEC_DIR,
            "embedding_matrix.npy",
        )

        word2idx_file = os.path.join(
            WORD2VEC_DIR,
            "word2idx.json",
        )

        if os.path.isfile(embedding_file) and os.path.isfile(word2idx_file):

            logger.info("Loading existing Word2Vec artifacts...")

            embedding_matrix = np.load(embedding_file)

            with open(word2idx_file, "r", encoding="utf-8") as f:
                word2idx = json.load(f)

            logger.info("Word2Vec artifacts loaded successfully.")

        else:

            logger.info("Training Word2Vec model...")

            embedding_matrix, word2idx = train_word2vec(
                df=train_df,
                text_column="clean_text",
                embedding_dim=embedding_dim,
                window=window,
                min_count=min_count,
                epochs=word2vec_epochs,
            )

            save_word2vec_artifacts(
                embedding_matrix,
                word2idx,
                output_dir=WORD2VEC_DIR,
            )

            logger.info("Word2Vec artifacts saved successfully.")

        # ---------------- DataLoaders ---------------- #

        _, train_loader = load_train_data(
            train_df["clean_text"],
            train_df["category"],
            batch_size,
            True,
            max_len,
            word2idx
        )

        _, valid_loader = load_train_data(
            valid_df["clean_text"],
            valid_df["category"],
            batch_size,
            False,
            max_len,
            word2idx
        )

        # ---------------- Build model ---------------- #

        num_classes = train_df["category"].nunique()

        model, criterion, optimizer, scaler = build_model(
            embedding_matrix=embedding_matrix,
            hidden_size=hidden_dim,
            hidden_size_2=hidden_dim_2,
            num_classes=num_classes,
            learning_rate=learning_rate,
            class_weights=class_weights,
            dropout = dropout,
        )

        # ---------------- Scheduler ---------------- #

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        # ---------------- Early stopping ---------------- #

        early_stopping = EarlyStopping(
            patience=3,
            path="best_bigru_model.pth",
        )

        # ---------------- Train ---------------- #

        history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            device=device,
            epochs=epochs,
        )

        logger.info(
            "Training completed successfully. Best validation accuracy: %.4f",
            history["best_accuracy"],
        )

    except Exception:
        logger.exception("Training pipeline failed.")
        raise


if __name__ == "__main__":
    main()
