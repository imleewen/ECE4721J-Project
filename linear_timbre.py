from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, RobustScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, rand, ceil, lit, when, expr
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MusicYearPredictionMLP") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .getOrCreate()

# 1. Load data
def load_avro(path):
    return spark.read.format("avro").load(path)

df = load_avro("file:///mnt/hgfs/p1team05/output_A2Z.avro")
df = df.filter((col("year") != 0) & col("year").isNotNull())

# Cache the filtered dataframe
df.cache()
print(f"Initial row count: {df.count()}")

# 2. Prepare features - explode array columns
# Create timbre features (90 columns)
for i in range(90):
    df = df.withColumn(f"timbre_{i}", col("segments_timbre").getItem(i))

feature_cols = [f"timbre_{i}" for i in range(90)]

# 3. Data Balancing Function (Fixed version)
def balance_data(spark_df, year_col, target_range=(1924, 2011), bin_size=5, samples_per_bin=1000):
    """Resample data to balance year distribution in Spark"""
    # Create bins
    bins = list(range(target_range[0], target_range[1] + bin_size, bin_size))
    
    # Assign each record to a bin
    bin_expr = "case "
    for i, bin_start in enumerate(bins[:-1]):
        bin_end = bins[i+1]
        bin_expr += f"when {year_col} >= {bin_start} and {year_col} < {bin_end} then {i} "
    bin_expr += f"else {len(bins)-1} end"
    
    df_with_bins = spark_df.withColumn("bin", expr(bin_expr))
    
    # For each bin, sample the desired number of records
    balanced_dfs = []
    for bin_num in range(len(bins)):
        bin_df = df_with_bins.filter(col("bin") == bin_num)
        bin_count = bin_df.count()
        
        if bin_count == 0:
            print(f"Warning: Bin {bin_num} ({bins[bin_num]}-{bins[bin_num+1] if bin_num < len(bins)-1 else bins[bin_num]}+) has no records")
            continue  # Skip empty bins
        
        if bin_count > samples_per_bin:
            # Sample without replacement
            sampled = bin_df.orderBy(rand()).limit(samples_per_bin)
        else:
            # Sample with replacement (oversample)
            fraction = samples_per_bin / bin_count
            sampled = bin_df.sample(withReplacement=True, fraction=fraction, seed=42)
            # Ensure we get exactly samples_per_bin
            sampled = sampled.orderBy(rand()).limit(samples_per_bin)
        
        balanced_dfs.append(sampled)
    
    if not balanced_dfs:
        raise ValueError("No data available in any bins for balancing")
    

    result_df = balanced_dfs[0]
    for df_to_union in balanced_dfs[1:]:
        result_df = result_df.unionByName(df_to_union)
    
    return result_df

# 4. Balance the dataset
try:
    balanced_df = balance_data(df, "year")
    balanced_df.cache()
    print(f"Balanced dataset count: {balanced_df.count()}")
except ValueError as e:
    print(f"Error in balancing data: {e}")
    spark.stop()
    exit(1)

# 5. Train-Test Split
train, test = balanced_df.randomSplit([0.8, 0.2], seed=42)

# Cache the training and test data
train.cache()
test.cache()
print(f"Training count: {train.count()}, Test count: {test.count()}")

# 6. Feature Scaling and Model Pipeline
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Use RobustScaler instead of StandardScaler
scaler = RobustScaler(inputCol="features", outputCol="scaledFeatures", 
                     withScaling=True, withCentering=True)

from pyspark.ml.feature import PolynomialExpansion

# After StandardScaler, add polynomial features
polyExpansion = PolynomialExpansion(
    degree=2, 
    inputCol="scaledFeatures", 
    outputCol="polyFeatures"
)

lr = LinearRegression(
    featuresCol="polyFeatures",
    labelCol="year",
    solver="l-bfgs",
    maxIter=400,
    tol=1e-6
)

# Create pipeline
pipeline = Pipeline(stages=[assembler, scaler, polyExpansion, lr])

# 7. Train model
model = pipeline.fit(train)

# 8. Make predictions
predictions = model.transform(test)

# Apply the same constraint logic (1924-2010)
predictions = predictions.withColumn(
    "constrained_prediction",
    when(col("prediction") < 1924, lit(1924))
    .when(col("prediction") > 2011, lit(2011))
    .otherwise(col("prediction"))
)

# Cache predictions
predictions.cache()

# 9. Evaluate
evaluator = RegressionEvaluator(
    labelCol="year",
    predictionCol="constrained_prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

# Calculate accuracy (exact year matches)
exact_matches = predictions.filter(col("year") == col("constrained_prediction")).count()
accuracy = exact_matches / predictions.count()

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f} years")
print(f"R^2: {r2:.2f}")
print(f"Exact year accuracy: {accuracy:.2%}")

# 10. Sample and plot results (convert to pandas for visualization)
pdf = predictions.select("year", "constrained_prediction").sample(0.1).toPandas()

plt.figure(figsize=(10, 5))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(pdf["year"], pdf["constrained_prediction"], alpha=0.3)
plt.plot([1924, 2011], [1924, 2011], '--r')
plt.xlabel('Actual Year')
plt.ylabel('Predicted Year')
plt.title(f'Actual vs Predicted (MAE: {mae:.1f} years)')

# Error Distribution
plt.subplot(1, 2, 2)
errors = pdf["year"] - pdf["constrained_prediction"]
plt.hist(errors, bins=30)
plt.xlabel('Prediction Error (years)')
plt.title('Error Distribution')

plt.tight_layout()
plt.show()

# 11. Clean up cached data
df.unpersist()
balanced_df.unpersist()
train.unpersist()
test.unpersist()
predictions.unpersist()

# Stop Spark session
spark.stop()