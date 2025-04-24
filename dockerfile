# Use Bitnami's Spark base image (with Hadoop)
FROM bitnami/spark:latest

# Set working directory
WORKDIR /app

# Copy Python scripts
COPY train_model.py predict_model.py run_model.py ./

# Copy datasets
COPY TrainingDataset.csv ValidationDataset.csv ./

# Optional: install common Python libraries (Spark handles most things, but in case you need anything extra)
RUN pip3 install pandas numpy

# Set environment variables for Spark to use Python 3
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Default command (can be overridden at runtime)
CMD ["spark-submit", "run_model.py", "TrainingDataset.csv", "ValidationDataset.csv"]
