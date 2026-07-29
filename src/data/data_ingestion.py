import pandas as pd
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

    
logger = logging.getLogger('data_ingestion')
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
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def processe_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data = data.dropna()
        data = data.drop_duplicates()

        data = data[data["clean_text"].str.strip() != ""]

        data["clean_text"] = (
            data["clean_text"]
            .str.lower()
            .str.strip()
            .str.replace(r"\n", " ", regex=True)
        )

        logger.debug("Data preprocessing completed")

        return data

    except Exception as e:
        logger.error("Unexpected error occurred while preprocessing the data: %s", e)
        raise
def save_data(
    data: pd.DataFrame,
    test_size: float,
    valid_size: float
) -> None:
    try:
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=42
        )

      
        train_data, valid_data = train_test_split(
            train_data,
            test_size=valid_size,
            random_state=42
        )

        os.makedirs("Data/raw", exist_ok=True)

        train_data.to_csv("Data/raw/train.csv", index=False)
        valid_data.to_csv("Data/raw/valid.csv", index=False)
        test_data.to_csv("Data/raw/test.csv", index=False)

        logger.info("Datasets saved successfully in Data/raw")

    except Exception:
        logger.exception("Failed to save datasets")
        
    
def main():
    try:

        params = load_param("params.yaml")
        test_size = params["data_ingestion"]["test_size"]
        valid_size = params["data_ingestion"]["valid_size"]
        data_path = "Data/clean_data.csv"
        logger.info("Loading dataset...")
        data = load_data(data_path)

        logger.info("Preprocessing dataset...")
        data = processe_data(data)

        logger.info("Saving train/validation/test datasets...")
        save_data(
            data=data,
            test_size=test_size,
            valid_size=valid_size
        )

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.exception("Data ingestion pipeline failed.")
        
if __name__ == "__main__":
    main()