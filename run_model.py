import sys
import os
from pyspark.sql import SparkSession
from train_model import train_model
from predict_model import predict_and_evaluate

def main():
    if len(sys.argv) != 4:
        print("Usage: run_model.py <train_dataset.csv> <validation_dataset.csv> <test_dataset.csv>")
        sys.exit(1)

    train_path = sys.argv[1]
    validation_path = sys.argv[2]
    test_path = sys.argv[3]

    # Check if files exist
    for path in [train_path, validation_path, test_path]:
        if not os.path.isfile(path):
            print(f"Error: File not found - {path}")
            sys.exit(1)

    spark = SparkSession.builder.appName("WineQualityPredictor").getOrCreate()

    print("Training model...")
    model = train_model(spark, train_path, validation_path)

    print("Evaluating model...")
    f1_score = predict_and_evaluate(spark, model, test_path)

    print(f"F1 Score on Test Dataset: {f1_score:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
