# === Stage 1: Clean the data ===
FROM bitnami/spark:latest AS cleaner

WORKDIR /app

# Optional: install common Python libraries (Spark handles most things, but in case you need anything extra)
RUN pip3 install pandas numpy

COPY debug_columns.py TrainingDataset.csv ValidationDataset.csv ./

# Clean datasets
RUN spark-submit debug_columns.py TrainingDataset.csv cleaned_train.csv && \
    spark-submit debug_columns.py ValidationDataset.csv cleaned_validation.csv

# === Stage 2: Final image with model files and cleaned data ===
FROM bitnami/spark:latest

WORKDIR /app

# Optional: install common Python libraries (Spark handles most things, but in case you need anything extra)
RUN pip3 install pandas numpy

# Copy cleaned data from the cleaner stage
COPY --from=cleaner /app/cleaned_train.csv ./cleaned_train.csv
COPY --from=cleaner /app/cleaned_validation.csv ./cleaned_validation.csv

# Copy training and prediction code
COPY train_model.py predict_model.py run_model.py ./

# Set the default command
CMD ["spark-submit", "train_model.py", "cleaned_train.csv", "cleaned_validation.csv"]
