from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
import numpy as np

# 1. Initialize Spark
spark = SparkSession.builder \
    .appName("YearPredictionWithTimbre") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .getOrCreate()

# 2. Load only needed columns
df = spark.read.format("avro").load("file:///home/hadoopuser/output_A.avro") \
    .select("segments_timbre", "year")

# 3. Remove invalid years
df = df.filter((col("year").isNotNull()) & (col("year") != 0))

# 4. Flatten segments_timbre array into separate columns (assume length 90)
for i in range(90):
    df = df.withColumn(f"timbre_{i}", col("segments_timbre")[i])

# 5. Select timbre columns + year
feature_cols = [f"timbre_{i}" for i in range(90)]
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

# 9. Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# 10. Use a regression model (GBTRegressor as MLPRegressor not available in regression)
regressor = GBTRegressor(
    labelCol="year_scaled",
    featuresCol="scaledFeatures",
    maxIter=100,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

# 11. Build pipeline
pipeline = Pipeline(stages=[assembler, scaler, regressor])

# 12. Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 13. Train
model = pipeline.fit(train_df)

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


# RMSE: 9.60
# MAE: 6.89 years
# R² Score: 0.20
