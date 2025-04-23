from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import os

if len(sys.argv) != 3:
    print("Usage: python predict_model.py <test_data_path> <model_path>")
    sys.exit(1)

test_path = sys.argv[1]
model_path = sys.argv[2]

if not os.path.exists(test_path) or not os.path.exists(model_path):
    print("Error: Input file or model path does not exist.")
    sys.exit(1)

spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load test data locally
test_df = spark.read.csv(test_path, header=True, inferSchema=True)

# Prepare features
feature_cols = [col for col in test_df.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
test_vec = assembler.transform(test_df).select("features", "quality")

# Load model
model = LogisticRegressionModel.load(model_path)

# Predict and evaluate
predictions = model.transform(test_vec)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Test F1 Score: {f1_score:.4f}")

spark.stop()
