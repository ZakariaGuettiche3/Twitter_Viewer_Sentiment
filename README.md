# Set up

pip install uv

uv init my_project_name

cd my_project_name

uv venv# Twitter Viewer Sentiment

An end-to-end sentiment analysis system built with a DVC-managed machine learning pipeline, MLflow experiment tracking and model registry, a Flask REST API for serving predictions, and a Chrome browser extension ("Comment Pulse") that surfaces the sentiment breakdown of a YouTube video's comments in real time.

Repository: https://github.com/ZakariaGuettiche3/Twitter_Viewer_Sentiment

---


## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Part 1 - AWS Setup](#part-1---aws-setup)
   - [1.1 Create an IAM User](#11-create-an-iam-user)
   - [1.2 Create an S3 Bucket](#12-create-an-s3-bucket)
   - [1.3 Launch the EC2 Instance](#13-launch-the-ec2-instance)
   - [1.4 Connect to and Configure the EC2 Instance](#14-connect-to-and-configure-the-ec2-instance)
   - [1.5 Point the Project at Your Tracking Server](#15-point-the-project-at-your-tracking-server)
6. [Part 2 - Local Project Setup](#part-2---local-project-setup)
7. [Part 3 - Running the Pipeline](#part-3---running-the-pipeline)
8. [Part 4 - Serving Predictions with Flask](#part-4---serving-predictions-with-flask)
9. [Part 5 - Installing the Browser Extension](#part-5---installing-the-browser-extension)
10. [Known Issues and Practical Notes](#known-issues-and-practical-notes)
11. [Security Notes](#security-notes)
12. [Tech Stack](#tech-stack)

---

## Project Overview

This project trains and serves a text sentiment classifier and exposes it through two consumer-facing surfaces:

- A machine learning pipeline, orchestrated with DVC, that ingests labeled text data, cleans and lemmatizes it, trains Word2Vec embeddings, and trains a Bidirectional GRU (BiGRU) classifier in PyTorch to predict whether a piece of text is negative, neutral, or positive.
- A Flask REST API that loads the trained model from the MLflow Model Registry and returns sentiment predictions, confidence scores, and distribution charts.
- A Chrome extension (Manifest V3), "Comment Pulse," that pulls a YouTube video's top-level comments through the YouTube Data API, sends them to the Flask API in batch, and renders the resulting sentiment split as charts inside the extension popup.

All experiment runs, metrics, and model versions are tracked and registered with MLflow, hosted on an AWS EC2 instance. Data and model artifacts are versioned with DVC, using AWS S3 as remote storage.

## Architecture

```
Data/clean_data.csv
        |
        v
 [DVC Pipeline]
 data_ingestion -> data_preprocessing -> model_build (Word2Vec + BiGRU) -> model_evaluation -> model_registration
        |                                                                        |
        v                                                                        v
 AWS S3 (DVC remote)                                              MLflow Tracking Server (AWS EC2)
                                                                     - experiment metrics
                                                                     - model registry (BGRU)
                                                                                 |
                                                                                 v
                                                                        Flask API (flask/app.py)
                                                                         /health /predict /predict_batch /chart
                                                                                 |
                                                                                 v
                                                              Chrome Extension "Comment Pulse"
                                                          (reads YouTube comments, calls Flask API,
                                                           renders sentiment split as charts)
```

## Repository Structure

```
Twitter_Viewer_Sentiment/
├── .dvc/                        DVC internal configuration
├── Data/                        Raw and processed datasets (DVC-tracked)
├── Notebooks/                   Exploratory notebooks
├── comment-pulse-extension/     Chrome extension (Manifest V3)
│   ├── manifest.json
│   ├── popup.html
│   └── popup.js
├── flask/                       Flask REST API for serving predictions
│   ├── app.py
│   └── preprocessing.py
├── models/                      Model registration metadata (model_info.json)
├── src/
│   ├── data/
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   └── model/
│       ├── model_build.py
│       ├── model_evaluation.py
│       └── model_register.py
├── dvc.yaml                     DVC pipeline definition
├── dvc.lock
├── params.yaml                  Pipeline hyperparameters
├── main.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

### Pipeline stages (defined in `dvc.yaml`)

1. **data_ingestion** — loads `Data/clean_data.csv`, drops duplicates and empty rows, splits into train/validation/test sets, and writes them to `Data/raw`.
2. **data_preprocessing** — lowercases text, strips special characters, removes stopwords, removes punctuation, and lemmatizes each sample; writes the result to `Data/preprocessed`.
3. **model_build** — trains Word2Vec embeddings on the training set and trains a BiGRU classifier in PyTorch, using early stopping and a learning-rate scheduler.
4. **model_evaluation** — evaluates the trained model on the held-out test set, logs parameters, metrics, and a confusion matrix to MLflow, and records the resulting model URI.
5. **model_registration** — registers the evaluated model under the name `BGRU` in the MLflow Model Registry and transitions it to the `Staging` stage.

## Prerequisites

- Python 3.11 or later
- Git
- `uv` (recommended) or `pip` for dependency management
- An AWS account with billing enabled
- Google Chrome (to load the browser extension)
- A Google Cloud project with the YouTube Data API v3 enabled and an API key (only required to use the browser extension — see [Part 5](#part-5---installing-the-browser-extension))

---

## Part 1 - AWS Setup

This section covers creating the AWS resources the project depends on: an IAM user for programmatic access, an S3 bucket used both as the DVC remote and the MLflow artifact store, and an EC2 instance that hosts the MLflow tracking server referenced throughout the codebase.

### 1.1 Create an IAM User

1. Sign in to the AWS Management Console with an account that has administrator access.
2. Open the **IAM** service and select **Users** in the left navigation, then **Create user**.
3. Enter a user name, for example `twitter-sentiment-dev`. Do not enable console access unless you specifically need it; this user is intended for programmatic (CLI/SDK) access.
4. On the permissions step, choose **Attach policies directly** and attach:
   - `AmazonS3FullAccess` — required for DVC to push/pull data and for MLflow to read and write model artifacts in S3.
   - `AmazonEC2FullAccess` — only needed if this same user will also be used to launch or manage the EC2 instance from the CLI. If you will create the EC2 instance manually through the console with your admin account, this policy is not required for the IAM user.

   For anything beyond local experimentation, replace these broad managed policies with a custom policy scoped to the specific S3 bucket created in the next step.

5. Review and create the user.
6. Open the new user, go to the **Security credentials** tab, and under **Access keys** select **Create access key**.
7. Choose **Command Line Interface (CLI)** as the use case, acknowledge the recommendation, and create the key.
8. Copy and store the **Access Key ID** and **Secret Access Key** immediately — the secret key is shown only once. Do not commit these to version control.

### 1.2 Create an S3 Bucket

The bucket serves two purposes in this project: DVC remote storage for datasets and model artifacts, and the MLflow artifact store for experiment runs.

1. Open the **S3** service and select **Create bucket**.
2. Choose a globally unique bucket name, for example `twitter-viewer-sentiment-artifacts`.
3. Choose an AWS Region and keep it consistent with the region you will use for the EC2 instance (the example MLflow tracking URI used in this codebase points to `eu-north-1`).
4. Leave **Block all public access** enabled.
5. Create the bucket.

### 1.3 Launch the EC2 Instance

The evaluation and model registration scripts (`src/model/model_evaluation.py`, `src/model/model_register.py`) and the Flask API (`flask/app.py`) all point to an MLflow tracking server reachable over HTTP on port 5000. This instance hosts that server.

1. Open the **EC2** service and select **Launch instance**.
2. Name the instance, for example `mlflow-tracking-server`.
3. Choose an Amazon Machine Image (AMI): Ubuntu Server 22.04 LTS or 24.04 LTS.
4. Choose an instance type: `t2.micro` or `t3.micro` is sufficient for a small tracking server and is Free Tier eligible.
5. Create a new key pair (for example `mlflow-key.pem`), download it, and store it securely. On Linux or macOS, restrict its permissions with `chmod 400 mlflow-key.pem`.
6. Under network settings, edit the security group and add the following inbound rules:
   - **SSH**, port 22, source restricted to your own IP address.
   - **Custom TCP**, port 5000, source restricted to your own IP address (or the IP ranges of any machines that need to reach the MLflow server, such as the machine running the Flask API). Avoid opening this to `0.0.0.0/0` outside of short-lived testing.
7. Leave the default storage (8-16 GB gp3 is enough).
8. Launch the instance.
9. Once running, note its **Public IPv4 DNS**, shown on the instance details page (format: `ec2-XX-XX-XX-XX.<region>.compute.amazonaws.com`). This value is what the MLflow tracking URI will point to.

### 1.4 Connect to and Configure the EC2 Instance

Connect over SSH from your local machine:

```bash
chmod 400 mlflow-key.pem
ssh -i mlflow-key.pem ubuntu@<your-ec2-public-dns>
```

Install system dependencies and set up a virtual environment:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv
python3 -m venv mlflow-env
source mlflow-env/bin/activate
pip install mlflow boto3 awscli
```

Configure AWS credentials on the instance so MLflow can read and write artifacts to the S3 bucket created earlier:

```bash
aws configure
# AWS Access Key ID:     <the IAM user's access key from step 1.1>
# AWS Secret Access Key: <the IAM user's secret access key from step 1.1>
# Default region name:   <your chosen region, e.g. eu-north-1>
# Default output format: json
```

Start the MLflow tracking server, using the S3 bucket as the artifact store. SQLite is used here as the backend store for simplicity; for anything beyond individual experimentation, use a managed database such as PostgreSQL instead.

```bash
nohup mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://<your-bucket-name>/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000 &
```

Running it with `nohup` (or, preferably, as a `systemd` service) keeps the server running after you disconnect from the SSH session. Once started, the MLflow UI is reachable at:

```
http://<your-ec2-public-dns>:5000/
```

### 1.5 Point the Project at Your Tracking Server

The MLflow tracking URI is currently hardcoded to a specific EC2 host in three files:

- `src/model/model_evaluation.py`
- `src/model/model_register.py`
- `flask/app.py`

Update the value passed to `mlflow.set_tracking_uri(...)` (and the `MLFLOW_TRACKING_URI` constant in `flask/app.py`) in each of these files to point at your own EC2 public DNS from step 1.3, for example:

```python
mlflow.set_tracking_uri("http://<your-ec2-public-dns>:5000/")
```

---

## Part 2 - Local Project Setup

### 2.1 Clone the repository

```bash
git clone https://github.com/ZakariaGuettiche3/Twitter_Viewer_Sentiment.git
cd Twitter_Viewer_Sentiment
```

### 2.2 Create a virtual environment and install dependencies

Using `uv` (as referenced in the project):

```bash
pip install uv
uv venv

# Windows
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate

uv add -r requirements.txt
```

Or with plain `pip`:

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2.3 Configure AWS credentials locally

```bash
aws configure
# Use the same IAM user access key and secret created in step 1.1
```

### 2.4 Configure the DVC remote

```bash
dvc init                # if not already initialized
dvc remote add -d storage s3://<your-bucket-name>/dvc-storage
dvc remote modify storage region <your-region>
```

To pull previously tracked data or model artifacts, if any exist in the remote:

```bash
dvc pull
```

### 2.5 Download the required NLTK data

The preprocessing stage uses NLTK's stopwords, tokenizer, part-of-speech tagger, and WordNet corpora:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

### 2.6 Provide the raw dataset

Place a labeled dataset at `Data/clean_data.csv` containing at minimum the following columns:

- `clean_text` — the raw text of the post or comment.
- `category` — the sentiment label (for example: negative, neutral, positive).

---

## Part 3 - Running the Pipeline

Create the log directory the pipeline scripts expect (see [Known Issues](#known-issues-and-practical-notes)):

```bash
mkdir -p log
```

Run the full DVC pipeline — data ingestion, preprocessing, Word2Vec and BiGRU training, evaluation, and MLflow registration:

```bash
dvc repro
```

View the pipeline graph:

```bash
dvc dag
```

Adjust hyperparameters (embedding size, GRU hidden dimensions, dropout, epochs, batch size, learning rate, and so on) in `params.yaml`, then re-run `dvc repro`. DVC re-executes only the stages whose dependencies or parameters changed.

Persist a completed pipeline run:

```bash
git add dvc.lock
git commit -m "Update pipeline run"
dvc push
```

---

## Part 4 - Serving Predictions with Flask

The Flask API additionally requires `flask-cors`, which is not currently listed in `requirements.txt` or `pyproject.toml`:

```bash
pip install flask-cors
```

Start the API:

```bash
cd flask
python app.py
```

The server starts on `http://0.0.0.0:5001`. It expects:

- A model already registered in MLflow at the configured `MODEL_URI` (`models:/BGRU/1` by default).
- The Word2Vec artifacts produced by the pipeline: `word2vec/word2idx.json` and `word2vec/embedding_matrix.npy`, located one directory above `flask/`.

### Endpoints

**`GET /health`**
Returns service status and the configured model URI.

**`POST /predict`**

```json
{ "text": "I really love this product!" }
```

Returns the predicted sentiment (negative, neutral, or positive), a confidence score, and a timestamp.

**`POST /predict_batch`**

```json
{
  "sentences": [
    { "text": "I love this!", "timestamp": "2026-07-29T10:00:00Z" },
    { "text": "This is bad." },
    "A plain string also works"
  ]
}
```

Returns a prediction for each item plus a base64-encoded PNG chart of the sentiment distribution.

**`POST /chart`**

```json
{ "results": [{ "sentiment": "positive" }, { "sentiment": "negative" }] }
```

Returns a standalone base64-encoded distribution chart built from an existing set of results.

---

## Part 5 - Installing the Browser Extension

The `comment-pulse-extension` folder contains a Manifest V3 Chrome extension, "Comment Pulse," that reads top-level comments from a YouTube video through the YouTube Data API, sends them to the Flask API's `/predict_batch` endpoint, and renders the resulting sentiment split as charts.

### 5.1 Obtain a YouTube Data API key

1. Open the Google Cloud Console and create or select a project.
2. Enable the **YouTube Data API v3** for that project.
3. Under **Credentials**, create an **API key** and copy it.

### 5.2 Load the extension

1. Make sure the Flask API is running locally on port 5001 (see [Part 4](#part-4---serving-predictions-with-flask)).
2. In Chrome, navigate to `chrome://extensions`.
3. Enable **Developer mode** (top right).
4. Click **Load unpacked** and select the `comment-pulse-extension` folder.
5. Open the extension's settings panel and enter your YouTube Data API key, and, if the backend is not running on the default `http://localhost:5001`, update the backend URL field accordingly.
6. Open a YouTube video and click the extension icon to view the sentiment breakdown of its comments.

Note: the extension's `manifest.json` grants host permissions only for `googleapis.com`, `localhost:5001`, and `127.0.0.1:5001`. If you deploy the Flask API elsewhere, update both the `host_permissions` in `manifest.json` and the backend URL used by the extension.

---

## Known Issues and Practical Notes

- **Missing `log/` directory**: `src/data/data_ingestion.py`, `src/data/data_preprocessing.py`, `src/model/model_build.py`, `src/model/model_evaluation.py`, and `src/model/model_register.py` all write error logs to a `log/` directory that is not created automatically. Run `mkdir -p log` before executing the pipeline or the scripts individually.
- **Case mismatch in `dvc.yaml`**: the `data_preprocessing` stage lists its dependencies as `data/raw/train.csv`, `data/raw/test.csv`, and `data/raw/valid.csv` (lowercase `data`), while `data_ingestion` writes its outputs to `Data/raw` (uppercase `Data`), and the scripts themselves consistently use `Data/`. On case-sensitive file systems (Linux, and most CI environments) this mismatch can cause `dvc repro` to fail to pick up the correct dependency; align the casing in `dvc.yaml` with the actual output path before relying on it in a fresh environment.
- **Missing `flask-cors` dependency**: `flask/app.py` imports `flask_cors`, which is not declared in `requirements.txt` or `pyproject.toml`. Install it manually as shown in [Part 4](#part-4---serving-predictions-with-flask).
- **Hardcoded MLflow tracking URI**: the tracking server address is hardcoded in three files rather than read from an environment variable. Update all three consistently if you point the project at your own MLflow server (see [1.5](#15-point-the-project-at-your-tracking-server)).

## Security Notes

- Never commit AWS access keys, the `.pem` key file, or the local `mlflow.db` file to version control.
- Restrict the EC2 security group's inbound rules to your own IP address rather than `0.0.0.0/0`, outside of short, deliberate testing windows.
- Once initial setup is complete, scope the IAM user's permissions down to the specific S3 bucket used by this project instead of leaving the broad `AmazonS3FullAccess` / `AmazonEC2FullAccess` managed policies attached long term.

## Tech Stack

- **Language**: Python 3.11+
- **Machine learning and NLP**: PyTorch, gensim (Word2Vec), scikit-learn, NLTK
- **Experiment tracking and model registry**: MLflow, hosted on AWS EC2
- **Data and pipeline versioning**: DVC, with AWS S3 as remote storage
- **Model serving**: Flask, Flask-CORS
- **Client**: Chrome extension (Manifest V3), Chart.js
- **Cloud infrastructure**: AWS (IAM, EC2, S3)


.venv\Scripts\activate.bat

uv add -r requirements.txt

# DCV

dvc init

dvc repro

dvc dag

# AWS

aws configure
