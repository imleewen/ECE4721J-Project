from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler, PCA
from pyspark.ml.regression import GBTRegressor, FMRegressor
from pyspark.ml import Pipeline
import numpy as np
import time


# 1. Initialize Spark
spark = SparkSession.builder \
    .appName("YearPredictionWithTimbre") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .getOrCreate()

# 2. Load only needed columns
df = spark.read.format("avro").load("file:///mnt/hgfs/p1team05/output_A.avro") \
    .select("segments_timbre", "segments_pitches", "year")

# 3. Remove invalid years
df = df.filter((col("year").isNotNull()) & (col("year") != 0))

# 4. Flatten segments_timbre array into separate columns (assume length 90)
for i in range(90):
    df = df.withColumn(f"timbre_{i}", col("segments_timbre")[i])
    df = df.withColumn(f"pitch_{i}", col("segments_pitches")[i])


# 5. Select timbre columns + year
feature_cols = [f"timbre_{i}" for i in range(90)] + [f"pitch_{i}" for i in range(90)]
df = df.select(feature_cols + ["year"])

# 6. Rescale year to [0, 1] (like sigmoid scaling)
min_year = 1924
max_year = 2011

scale_year_udf = udf(lambda y: float((y - min_year) / (max_year - min_year)), DoubleType())
df = df.withColumn("year_scaled", scale_year_udf(col("year")))

# 7. Remove nulls
df = df.dropna()

# 8. Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_df = assembler.transform(df)

# 9. Standardize features
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# First fit the scaler to get scaled features
scaler_model = scaler.fit(assembled_df)  # Use assembled_df instead of df
scaled_df = scaler_model.transform(assembled_df)

# Compute PCA to find optimal k
pca = PCA(k=180, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_model = pca.fit(scaled_df)

# Calculate cumulative explained variance
variances = pca_model.explainedVariance.toArray()
cumulative_var = np.cumsum(variances)
k = np.argmax(cumulative_var >= 0.9) + 1  # +1 because Python is 0-indexed

print(f"Number of components explaining 95% variance: {k}")

# Now create final PCA with optimal k
optimal_pca = PCA(k=k, inputCol="scaledFeatures", outputCol="pcaFeatures")

# 10. Update regressor to use PCA features
regressor = FMRegressor(
    labelCol="year_scaled",
    featuresCol="pcaFeatures",  # Changed from scaledFeatures
    factorSize=4,
    solver="adamW",
    miniBatchFraction=0.2,
    maxIter=400,
    stepSize=0.05,
    regParam=0.01,
    seed=42
)

# with PCA
# RMSE: 10.58                                                                     
# MAE: 7.82 years
# R² Score: 0.06
# Time taken: 185.85


# 11. Build pipeline
pipeline = Pipeline(stages=[assembler, scaler, optimal_pca, regressor])

# 12. Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
start = time.time()

# 13. Train
model = pipeline.fit(train_df)

duration = time.time() - start

# 14. Predict
predictions = model.transform(test_df)

# 15. Inverse scale prediction
inverse_scale_udf = udf(lambda y: float(y * (max_year - min_year) + min_year), DoubleType())
predictions = predictions.withColumn("year_pred", inverse_scale_udf(col("prediction")))

# 16. Round prediction to integer
from pyspark.sql.functions import ceil
predictions = predictions.withColumn("year_pred_int", ceil(col("year_pred")).cast("int"))

# 17. Evaluation
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="year", predictionCol="year_pred", metricName="rmse")
rmse = evaluator.evaluate(predictions)

evaluator_mae = RegressionEvaluator(labelCol="year", predictionCol="year_pred", metricName="mae")
mae = evaluator_mae.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="year", predictionCol="year_pred", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f} years")
print(f"R² Score: {r2:.2f}")
print(f"Time taken: {duration:.2f}")
print(f"Number of components explaining 95% variance: {k}")

import matplotlib.pyplot as plt
pdf = predictions.select("year", "prediction").sample(0.1).toPandas()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pdf["year"], pdf["prediction"], alpha=0.3)
plt.plot([1924, 2011], [1924, 2011], '--r')
plt.xlabel("Actual Year")
plt.ylabel("Predicted Year")
plt.title("Actual vs Predicted Years")

plt.subplot(1, 2, 2)
errors = pdf["year"] - pdf["prediction"]
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error (years)")
plt.title("Error Distribution")

plt.tight_layout()
plt.show()

# Stop Spark session
spark.stop()