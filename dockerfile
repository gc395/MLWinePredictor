# Use Spark-enabled Python image
FROM bitnami/spark:latest

WORKDIR /app

# Copy scripts
COPY train_model.py predict_model.py debug_columns.py run_model.py ./

# Copy raw datasets
COPY TrainingDataset.csv ValidationDataset.csv ./

# Optional: install common Python libraries (Spark handles most things, but in case you need anything extra)
RUN pip3 install pandas numpy

# Set Python version for Spark
ENV PYSPARK_PYTHON=python3

# Run cleanup + training at container startup
CMD ["sh", "-c", "\
    spark-submit debug_columns.py TrainingDataset.csv cleaned_train.csv && \
    spark-submit debug_columns.py ValidationDataset.csv cleaned_validation.csv && \
    spark-submit train_model.py cleaned_train.csv cleaned_validation.csv && \
    spark-submit run_model.py cleaned_train.csv cleaned_validation.csv | tee /app/output.txt"]

