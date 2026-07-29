import pandas as pd
import os
import logging
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
nltk.download("stopwords")

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('log/errors_preprocessing.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_text(text):
    words = nltk.word_tokenize(str(text))
    pos_tags = nltk.pos_tag(words)

    return " ".join(
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    )


def load_data(data_url: str) -> pd.DataFrame:
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


def preprocess_data(
    data: pd.DataFrame,
    text_column: str = "clean_text"
) -> pd.DataFrame:

    try:
        logger.info("Starting text preprocessing...")

        data = data.copy()

        logger.debug("Initial dataset shape: %s", data.shape)

        # Remove empty texts
        data = data[data[text_column].str.strip() != ""]
        logger.debug("Removed empty texts. Shape: %s", data.shape)

        # Lowercase
        data[text_column] = data[text_column].str.lower()

        # Remove leading/trailing spaces
        data[text_column] = data[text_column].str.strip()

        # Remove special characters
        data[text_column] = data[text_column].str.replace(
            r"[^a-zA-Z0-9\s]",
            "",
            regex=True
        )

        # Replace new lines
        data[text_column] = data[text_column].str.replace(
            r"\n",
            " ",
            regex=True
        )

        logger.debug("Basic text cleaning completed.")

        # Load stopwords
        stop_words = set(stopwords.words("english"))
        stop_words -= {"not", "no", "but"}

        # Remove stopwords
        data[text_column] = data[text_column].apply(
            lambda text: " ".join(
                word
                for word in str(text).split()
                if word not in stop_words
            )
        )

        logger.debug("Stopwords removed.")

        data[text_column] = data[text_column].apply(
            lambda text: text.translate(
                str.maketrans("", "", string.punctuation)
            )
        )

        logger.debug("Punctuation removed.")

        data[text_column] = data[text_column].apply(lemmatize_text)

        logger.debug("Lemmatization completed.")

        logger.info("Text preprocessing completed successfully.")

        return data

    except KeyError:
        logger.exception("Column '%s' does not exist in the DataFrame.", text_column)
        raise

    except Exception:
        logger.exception("Unexpected error during text preprocessing.")
        raise
    
def save_data(
    data: pd.DataFrame,
    name: string,
) -> None:
    try:
       
        os.makedirs("Data/preprocessed", exist_ok=True)

        data.to_csv(f"Data/preprocessed/{name}_preprocessed.csv", index=False)

        logger.info("Datasets saved successfully in Data/preprocessed")

    except Exception:
        logger.exception("Failed to save datasets")
        
def main():
    try:
        logger.info("Starting data preprocessing pipeline...")

        logger.info("Loading training dataset...")
        train_data = load_data("Data/raw/train.csv")

        logger.info("Loading validation dataset...")
        valid_data = load_data("Data/raw/valid.csv")

        logger.info("Loading test dataset...")
        test_data = load_data("Data/raw/test.csv")

  
        logger.info("Preprocessing training dataset...")
        train_data = preprocess_data(train_data)

        logger.info("Preprocessing validation dataset...")
        valid_data = preprocess_data(valid_data)

        logger.info("Preprocessing test dataset...")
        test_data = preprocess_data(test_data)

   
        logger.info("Saving training dataset...")
        save_data(train_data, "train")

        logger.info("Saving validation dataset...")
        save_data(valid_data, "valid")

        logger.info("Saving test dataset...")
        save_data(test_data, "test")

        logger.info("Data preprocessing pipeline completed successfully.")

    except Exception:
        logger.exception("Data preprocessing pipeline failed.")
        raise


if __name__ == "__main__":
    main()
