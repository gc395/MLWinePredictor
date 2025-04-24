# MLWinePredictor

A machine learning pipeline for predicting wine quality using PySpark and MLlib, containerized with Docker for portability and easy deployment. Built for use in distributed computing environments like AWS EMR, but works locally as well.

## Overview

This project uses the `TrainingDataset.csv` and `ValidationDataset.csv` to train and validate a **Logistic Regression** model that predicts the quality of wine (on a scale of 1 to 10). The pipeline is designed to be modular, debuggable, and scalable. The final model performance is reported using **F1 Score**, as per project requirements.

---

## Features

- ✅ Built using **Apache Spark** and **PySpark**
- ✅ Uses **MLlib** for machine learning (Logistic Regression)
- ✅ Data cleaning step for malformed CSV headers
- ✅ Automatically trains and evaluates model
- ✅ Runs entirely inside Docker
- ✅ Output F1 score saved and viewable
- ✅ Easily replace model with other classifiers (e.g., Random Forest)

---

## Project Structure

```
.
├── debug_columns.py         # Cleans and reformats column names
├── train_model.py           # Trains the MLlib Logistic Regression model
├── predict_model.py         # Predicts & evaluates using the trained model
├── run_model.py             # Controls overall execution flow
├── TrainingDataset.csv      # Raw semicolon-delimited training dataset
├── ValidationDataset.csv    # Raw semicolon-delimited validation dataset
├── Dockerfile               # Docker setup for full Spark pipeline
└── output.txt               # Output file with printed F1 Score (after run)
```

---

## Requirements

- Docker (Linux/amd64 platform recommended)
- Training & Validation CSVs in root directory
- Compatibility with Spark on AWS EMR

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/gc395/MLWinePredictor.git
cd MLWinePredictor
```

### 2. Add Datasets (if not already available)

Place the following files in the project directory:

- `TrainingDataset.csv`
- `ValidationDataset.csv`

Ensure they are semicolon (`;`) delimited.

### 3. Build the Docker Image

The docker image is already on this repo; however, if you prefer to pull the image from Docker Hub:

```bash
docker pull gcancino98/ml-wine-predictor
```

Build the image by running the following command:

```bash
docker build -t ml-wine-predictor .
```

### 4. Run the Container

```bash
docker run --name <container_id> gcancino98/ml-wine-predictor:latest
```

```<container_id>``` can be any name of identifier. Make note of this as you'll need it to grab the output.

### 5. Get the Output

```bash
docker cp wine-cleaner:/app/output.txt ./output.txt
```

This will copy the file to your current directory. You can specify another directory if you like by changing the last parameter.

### 6. Get the F1 Score

Using the ```output.txt``` file that you copied from the docker container, you can run ```bash grep "F1 Score" <directory/where/you/copied/the/output/./txt>```

## Example Output

By grep-ing `output.txt`, you’ll find (example):

```
Validation F1 Score: 0.5610
```

> Your score may vary depending on the data and model used.

---

## Swapping the Model (Optional)

To try **Random Forest** or other classifiers, simply edit `train_model.py`:

```python
from pyspark.ml.classification import RandomForestClassifier

model = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=100)
```
Build the docker image before running the container

---

## Debugging

- Check CSV format (must be `;` delimited; ```debug_columns.py already handles this, but just in case)
- Ensure Docker is using the correct architecture (`linux/amd64`)*
- Use `docker logs <container_id>` for container output

---

## Cleanup

Remove all containers/images after testing:

```bash
docker container prune
docker image prune -a
```

---

## Compatibility between M1/2/3/4 MacBooks and EMR

EMR is classified as an AMD64 machine, whereas MacBooks with an M processor are classified as ARM64. This difference in architecture will result in a given image not working for the other architecture, specifically ARM64 on a AMD64 machine. Building an image on an M chip MacBook and running it on an AMD64 EMR instance will throw an error referring to the incorrect context. 
To fix this, you'll need to either build the docker image inside the EC2 instance (which is already stated in this README) or explicitly define the platform type with ```--platform amd64``` and push it to Docker Hub. 

Feel free to raise issues or reach out on [GitHub](https://github.com/gc395/MLWinePredictor/issues).
