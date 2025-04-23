# Use an official Python base with Spark pre-installed
FROM bitnami/spark:latest

# Set working directory
WORKDIR /app

# Copy Python scripts
COPY train_model.py predict_model.py ./

# Copy training and validation datasets into the container
COPY TrainingDataset.csv ValidationDataset.csv ./

# Set environment variable to suppress Spark's interactive shell prompt
ENV PYSPARK_PYTHON=python3

# Default command (can be overridden)
CMD ["spark-submit", "train_model.py", "TrainingDataset.csv", "ValidationDataset.csv"]
