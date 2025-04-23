FROM bitnami/spark:3.3.2

USER root

# Install Python packages (optional: add more as needed)
RUN pip install --upgrade pip && \
    pip install pandas

# Copy app files
COPY train_model.py /app/train_model.py
COPY predict_model.py /app/predict_model.py

WORKDIR /app

ENTRYPOINT ["spark-submit"]
