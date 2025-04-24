FROM bitnami/spark:latest

WORKDIR /app

COPY train_model.py predict_model.py debug_columns.py run_model.py ./
COPY TrainingDataset.csv ValidationDataset.csv ./

ENV PYSPARK_PYTHON=python3

RUN pip3 install pandas numpy

CMD ["sh", "-c", "\
    spark-submit debug_columns.py TrainingDataset.csv cleaned_train && \
    spark-submit debug_columns.py ValidationDataset.csv cleaned_validation && \
    spark-submit run_model.py cleaned_train cleaned_validation | tee /app/output.txt"]
