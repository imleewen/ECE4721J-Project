from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F

# Initialize Spark
spark = SparkSession.builder \
    .appName("PCA + Linear Regression") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .getOrCreate()

# Load data
df = spark.read.format("avro").load("file:///home/hadoopuser/p1team05/combined.avro")

# Extract pitch and timbre columns
for i in range(12):
    df = df.withColumn(f"pitch_mean_{i}", F.col("segments_pitches_mean")[i])
    df = df.withColumn(f"pitch_std_{i}", F.col("segments_pitches_std")[i])
    df = df.withColumn(f"timbre_mean_{i}", F.col("segments_timbre_mean")[i])
    df = df.withColumn(f"timbre_std_{i}", F.col("segments_timbre_std")[i])

# Select numeric columns (exclude 'year')
numeric_cols = [
    field.name for field in df.schema.fields
    if isinstance(field.dataType, (IntegerType, DoubleType, FloatType, LongType)) and field.name != "year"
]

# Drop nulls and invalid years
df = df.na.drop(subset=numeric_cols + ["year"]).filter(F.col("year") != 0)

# Assemble features
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

# Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# PCA with top 32 components
pca = PCA(k=32, inputCol="scaledFeatures", outputCol="pcaFeatures")

# Define pipeline
pipeline = Pipeline(stages=[assembler, scaler, pca])

# Fit and transform
model = pipeline.fit(df)
pca_result = model.transform(df)

# Train/test split
train_data, test_data = pca_result.randomSplit([0.8, 0.2], seed=42)

# ================= Linear Regression =================
lr = LinearRegression(featuresCol="pcaFeatures", labelCol="year")
lr_model = lr.fit(train_data)
predictions = lr_model.transform(test_data)

# ================= Evaluation =================
evaluator = RegressionEvaluator(labelCol="year", predictionCol="prediction")

rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R^2: {r2:.2f}")
print(f"MAE: {mae:.2f}")

