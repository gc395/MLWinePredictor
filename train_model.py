from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import sys

def main(train_path, val_path, model_output_path):
    spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

    # Load datasets
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)
    val_df = spark.read.csv(val_path, header=True, inferSchema=True)

    # Preprocessing
    features = [col for col in train_df.columns if col != "quality"]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    # Model
    lr = LogisticRegression(maxIter=50, regParam=0.3, elasticNetParam=0.8)

    pipeline = Pipeline(stages=[assembler, label_indexer, lr])
    model = pipeline.fit(train_df)

    # Validation
    predictions = model.transform(val_df)
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    print(f"Validation F1 Score: {f1_score:.4f}")

    # Save model
    model.write().overwrite().save(model_output_path)
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: train_model.py <train_path> <val_path> <model_output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
