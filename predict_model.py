from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import sys

def main(model_path, test_path, output_path):
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Load model and data
    model = PipelineModel.load(model_path)
    test_df = spark.read.csv(test_path, header=True, inferSchema=True)

    # Predict
    predictions = model.transform(test_df)
    predictions.select("prediction").write.csv(output_path, header=True, mode="overwrite")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: predict_model.py <model_path> <test_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
