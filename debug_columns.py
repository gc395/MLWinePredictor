from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder \
    .appName("DebugWine") \
    .getOrCreate()

# Read CSV from HDFS with proper options
df = spark.read \
    .option("header", True) \
    .option("delimiter", ";") \
    .option("quote", '"') \
    .csv("hdfs:///user/hadoop/new_input_data.csv")


# Strip out extra quotes and whitespace from column names
clean_cols = [col.strip().replace('"""', '').replace('""', '').replace('"', '') for col in df.columns]
df = df.toDF(*clean_cols)

# Print cleaned column names
print("=== Cleaned Columns ===")
for idx, col_name in enumerate(df.columns):
    print(f"{idx}: '{col_name}'")

# Optional: show a few rows to verify
df.show(5)

# Stop the Spark session
spark.stop()
