from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

# Initialize Spark
spark = SparkSession.builder \
    .appName("PCA") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:/home/hadoopuser/spark/conf/log4j2.properties") \
    .getOrCreate()

# Load data
df = spark.read.format("avro").load("file:///home/hadoopuser/p1team05/combined.avro")

# For segments_pitches (length 12 array)
for i in range(12):
    df = df.withColumn(f"pitch_mean_{i}", F.col("segments_pitches_mean")[i])
    df = df.withColumn(f"pitch_std_{i}", F.col("segments_pitches_std")[i])

# For segments_timbre (length 12 array)
for i in range(12):
    df = df.withColumn(f"timbre_mean_{i}", F.col("segments_timbre_mean")[i])
    df = df.withColumn(f"timbre_std_{i}", F.col("segments_timbre_std")[i])

# Select numeric columns but exclude 'year'
numeric_cols = [
    field.name for field in df.schema.fields 
    if isinstance(field.dataType, (IntegerType, DoubleType, FloatType, LongType))
    and field.name != "year"  # Exclude the year column
]

# Remove any rows with null values and year = 0
df = df.na.drop(subset=numeric_cols + ["year"]).filter(F.col("year") != 0)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

# Standardize the features (important for PCA)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# # Fit pipeline (without PCA)
# pipeline = Pipeline(stages=[assembler, scaler])
# model = pipeline.fit(df)
# result = model.transform(df)

# Principal Component Analysis
pca = PCA(k=len(numeric_cols), inputCol="scaledFeatures", outputCol="pcaFeatures")

# Fit pipeline
pipeline = Pipeline(stages=[assembler, scaler, pca])
model = pipeline.fit(df)
pca_result = model.transform(df)

# Split data into train/test
train_data, test_data = pca_result.randomSplit([0.8, 0.2], seed=42)
# train_data, test_data = result.randomSplit([0.8, 0.2], seed=42)

# ================================================================================
#                            Gradient-Boosted Trees
# ================================================================================

from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol="pcaFeatures", labelCol="year")
gbt_model = gbt.fit(train_data)
predictions = gbt_model.transform(test_data)

# ================================================================================
#                                   Evaluation
# ================================================================================

from pyspark.ml.evaluation import RegressionEvaluator

# Define evaluator
evaluator = RegressionEvaluator(labelCol="year", predictionCol="prediction")

# Available metrics: "rmse", "mse", "r2", "mae"
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

print(f"RMSE: {rmse}")  # Root Mean Squared Error
print(f"MSE: {mse}")    # Mean Squared Error (more sensitive to outliers)
print(f"R^2: {r2}")     # R-squared
print(f"MAE: {mae}")    # Mean Absolute Error (robust to outliers)