# Use Bitnami's Spark base image (with Hadoop)
FROM bitnami/spark:latest

# Set working directory
WORKDIR /app

# Copy Python scripts
COPY train_model.py predict_model.py run_model.py ./

# Copy datasets
COPY TrainingDataset.csv ValidationDataset.csv ./

# Set environment variable so Spark uses Python 3
ENV PYSPARK_PYTHON=python3

# Optionally install extra Python packages (if needed)
# USER root
# RUN apt-get update && apt-get install -y python3-pip && \
#     pip3 install pandas scikit-learn

# Switch back if you elevated privileges
# USER 1001

# Default command (can be overridden at runtime)
CMD ["spark-submit", "run_model.py", "TrainingDataset.csv", "ValidationDataset.csv"]
