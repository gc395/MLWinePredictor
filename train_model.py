from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import os

if len(sys.argv) != 3:
    print("Usage: python train_model.py <training_data_path> <validation_data_path>")
    sys.exit(1)

training_path = sys.argv[1]
validation_path = sys.argv[2]

if not os.path.exists(training_path) or not os.path.exists(validation_path):
    print("Error: One or both input paths do not exist.")
    sys.exit(1)

spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

train_df = spark.read.csv(training_path, header=True, inferSchema=True)
val_df = spark.read.csv(validation_path, header=True, inferSchema=True)

feature_cols = [col for col in train_df.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_vec = assembler.transform(train_df).select("features", "quality")
val_vec = assembler.transform(val_df).select("features", "quality")

lr = LogisticRegression(labelCol="quality", featuresCol="features", maxIter=10)
model = lr.fit(train_vec)

model.write().overwrite().save("trained_model")

predictions = model.transform(val_vec)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score:.4f}")

spark.stop()
