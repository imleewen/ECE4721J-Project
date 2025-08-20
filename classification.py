from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import StorageLevel
import pyspark.sql.functions as F
import numpy as np

# Config
THRESHOLD = 0.99          # Target cumulative explained variance
MIN_STD = 1e-8            # Minimum std threshold to filter out near-constant features
MAX_K = 70               # Max number of components to try in PCA

# Initialize Spark
spark = SparkSession.builder \
    .appName("Optimized_NN") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:/home/hadoopuser/spark/conf/log4j2.properties") \
    .config("spark.executor.instances", "4") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.default.parallelism", "100") \
    .config("spark.ui.showConsoleProgress", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.executor.extraJavaOptions", 
       "-XX:+UseG1GC "
       "-XX:G1HeapRegionSize=16m "
       "-XX:InitiatingHeapOccupancyPercent=35 "
       "-XX:ConcGCThreads=4 "
       "-XX:ParallelGCThreads=4 "
       "-XX:+UnlockExperimentalVMOptions "
       "-XX:G1NewSizePercent=20 "
       "-XX:G1MaxNewSizePercent=40") \
    .getOrCreate()

# Load and prepare data
df = spark.read.format("avro").load("file:///mnt/hgfs/p1team05/combined.avro")

# Expand array columns
for i in range(12):
    df = df.withColumn(f"pitch_mean_{i}", F.col("segments_pitches_mean")[i]) \
           .withColumn(f"pitch_std_{i}", F.col("segments_pitches_std")[i]) \
           .withColumn(f"timbre_mean_{i}", F.col("segments_timbre_mean")[i]) \
           .withColumn(f"timbre_std_{i}", F.col("segments_timbre_std")[i])

# Select numeric features
raw_numeric = [
    field.name for field in df.schema.fields 
    if isinstance(field.dataType, (IntegerType, DoubleType, FloatType, LongType))
    and field.name != "year"
]

# Remove irrelevant features (e.g., IDs, sample_rate, etc.)
bad_tokens = ("id", "playmeid", "sample_rate", "7digital")
numeric_cols = [c for c in raw_numeric if not any(tok in c.lower() for tok in bad_tokens)]

print("=" * 60)
print("Initial numeric cols:", len(raw_numeric))
print("Filtered numeric cols:", len(numeric_cols))
print("=" * 60)

# Clean data - drop nulls and rows with year == 0
df = df.na.drop(subset=numeric_cols + ["year"]).filter(F.col("year") != 0)

# Remove low-variance features
stats = df.select([F.stddev(F.col(c)).alias(c) for c in numeric_cols]).collect()[0].asDict()
low_var = [c for c, s in stats.items() if (s is None) or (s < MIN_STD)]
numeric_cols = [c for c in numeric_cols if c not in low_var]

print("Dropped low-variance cols:", len(low_var))
if low_var:
    print("Examples:", low_var[:10])

if not numeric_cols:
    raise ValueError("No usable features left after filtering. Check your data.")

print("Final numeric cols used:", len(numeric_cols))
print("=" * 60)

# Assemble & scale features
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

# First run PCA to determine optimal components
k = min(MAX_K, len(numeric_cols))
pca = PCA(k=k, inputCol="scaledFeatures", outputCol="pcaFeatures")

# Fit PCA to determine optimal number of components
pipeline = Pipeline(stages=[assembler, scaler, pca])
model = pipeline.fit(df)
pca_model = model.stages[-1]

# Explained Variance Output
explained = pca_model.explainedVariance.toArray()
cum = explained.cumsum()

print("\nExplained variance by component:")
for i, (e, c) in enumerate(zip(explained, cum), 1):
    print(f"PC{i:02d}: explained = {e:.4f}, cumulative = {c:.4f}")

# Determine how many components are needed
mask = cum >= THRESHOLD
if mask.any():
    optimal_k = int(np.argmax(mask) + 1)
    print(f"\nComponents needed to reach >= {int(THRESHOLD*100)}% percent variance: {optimal_k}")
else:
    optimal_k = len(explained)
    print(f"\nCould not reach {int(THRESHOLD*100)}% variance. Total explained: {cum[-1]*100:.2f}% "
          f"using {len(explained)} components.")

# Now create the final pipeline with optimal PCA components
final_pca = PCA(k=optimal_k, inputCol="scaledFeatures", outputCol="pcaFeatures")
indexer = StringIndexer(inputCol="year", outputCol="label")

# Full preprocessing pipeline with optimal PCA
full_pipeline = Pipeline(stages=[assembler, scaler, final_pca, indexer])
preprocessed_data = full_pipeline.fit(df).transform(df) \
    .select("pcaFeatures", "label") \
    .persist(StorageLevel.MEMORY_AND_DISK)

preprocessed_data.count()  # Force caching

# Get number of classes
num_classes = preprocessed_data.select("label").distinct().count()
print(f"Number of classes (years): {num_classes}")

# Train/test split
train_data, test_data = preprocessed_data.randomSplit([0.8, 0.2], seed=42)
# Train/test split with stratification
# fractions = {label: 0.6 for label in range(num_classes)}
# train_data = preprocessed_data.sampleBy("label", fractions, seed=42)
# test_data = preprocessed_data.subtract(train_data)

import time

# Force materialization with garbage collection
for _ in range(3):
    train_data.count()
    time.sleep(10)  # Allow GC between ops

# Neural Network - adjust input layer size based on PCA components
mlp = MultilayerPerceptronClassifier(
    layers=[optimal_k, 64, 32, num_classes],  # First layer matches PCA components
    featuresCol="pcaFeatures",
    labelCol="label",
    solver="l-bfgs",
    blockSize=1024,
    maxIter=200,
    tol=1e-6,
    seed=42,
    stepSize=0.03
)

# Train model
mlp_model = mlp.fit(train_data)

# Evaluate
predictions = mlp_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Show predictions
predictions.groupBy("label", "prediction").count().show(20)

spark.stop()