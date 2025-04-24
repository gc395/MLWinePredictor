import sys
from pyspark.sql import SparkSession

# Validate input and output arguments
if len(sys.argv) != 3:
    print("Usage: debug_columns.py <input_csv> <output_csv>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Start Spark session
spark = SparkSession.builder.appName("DebugWine").getOrCreate()

# Read CSV with proper options
df = spark.read \
    .option("header", True) \
    .option("delimiter", ";") \
    .option("quote", '"') \
    .csv(input_path)

# Clean column names
clean_cols = [col.strip().replace('"""', '').replace('""', '').replace('"', '') for col in df.columns]
df = df.toDF(*clean_cols)

# Save cleaned version
df.write.mode("overwrite").option("header", True).csv(output_path)

spark.stop()
