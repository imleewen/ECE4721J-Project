from pyspark.sql import SparkSession
from pyspark.sql.functions import count, min ,max, col
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType

# Initialize Spark
spark = SparkSession.builder \
    .appName("MusicYearPrediction") \
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

# Load data
df = spark.read.format("avro").load("file:///mnt/hgfs/p1team05/output_A.avro")

# ================================================================================
#                                Data Exploration
# ================================================================================

df = df.na.drop(subset=["segments_timbre", "segments_pitches", "year"]).filter(F.col("year") != 0)

# Check year range and distribution
year_stats = df.agg(
    min("year").alias("min_year"),
    max("year").alias("max_year"),
    count("year").alias("total_records")
).collect()

min_year = year_stats[0]["min_year"]
max_year = year_stats[0]["max_year"]
total_records = year_stats[0]["total_records"]

print(f"Year Range: {min_year} to {max_year}")
print(f"Total records: {total_records}")

# Convert to decades (reduces from 78 classes to 9)
df = df.withColumn("decade", (F.floor(col("year")/10)*10).cast("int"))

# New label mapping (1920s=0, 1930s=1,..., 2010s=8)
decade_mapping = df.select("decade").distinct().orderBy("decade") \
    .withColumn("decade_label", (col("decade")-1920)/10)
df = df.join(decade_mapping, "decade")

from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.linalg import VectorUDT

# Define UDF to convert array to dense vector
def array_to_vector(arr):
    return Vectors.dense(arr)

# Register UDF with proper return type
array_to_vector_udf = udf(array_to_vector, VectorUDT())

# Convert array columns to vector columns
df = df.withColumn("timbre_vector", array_to_vector_udf(col("segments_timbre"))) \
       .withColumn("pitches_vector", array_to_vector_udf(col("segments_pitches")))

# Now we can use VectorAssembler with the vector columns
assembler = VectorAssembler(
    inputCols=["timbre_vector", "pitches_vector"],
    outputCol="combined_features"
)

scaler = StandardScaler(
    inputCol="combined_features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

pca = PCA(
    k=45, # cumulative variance = 0.9903
    inputCol="scaled_features",
    outputCol="pca_features"
)

feature_pipeline = Pipeline(stages=[assembler, scaler])
feature_model = feature_pipeline.fit(df)
transformed_df = feature_model.transform(df)

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

# Create sequential labels (0 to N-1) only for existing years
year_mapping = transformed_df.select("decade").distinct() \
    .orderBy("decade") \
    .withColumn("label", row_number().over(Window.orderBy("decade")) - 1)

# Join back to original data
transformed_df = transformed_df.join(year_mapping, "decade")

# Split data
train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

# Get number of classes (unique years)
num_classes = year_mapping.count()

# Check for years in test set not in training
test_years = test_df.select("decade").distinct()
train_years = train_df.select("decade").distinct()
missing_years = test_years.subtract(train_years)
print(f"Decades in test not in train: {missing_years.count()}")

from pyspark.ml.classification import RandomForestClassifier

# Calculate class weights
class_weights = train_df.groupBy("label").count().rdd.collectAsMap()
total = train_df.count()
weights = {k: total/(len(class_weights)*v) for k,v in class_weights.items()}

# Add weight column
weight_udf = F.udf(lambda label: weights[label], FloatType())
train_df = train_df.withColumn("weight", weight_udf(col("label")))
test_df = test_df.withColumn("weight", weight_udf(col("label")))

# Update RF with weights
rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    weightCol="weight",
    numTrees=300,  # Increased from default
    maxDepth=15,   # Deeper trees
    minInstancesPerNode=5,
    subsamplingRate=0.7
)

rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

# Update evaluator for classification
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# Evaluate
mlp_f1 = evaluator.evaluate(rf_predictions)
mlp_accuracy = evaluator.setMetricName("accuracy").evaluate(rf_predictions)
mlp_precision = evaluator.setMetricName("weightedPrecision").evaluate(rf_predictions)
mlp_recall = evaluator.setMetricName("weightedRecall").evaluate(rf_predictions)

print("\nRF Classifier Results:")
# print(f"Best parameters: {best_model.extractParamMap()}")
print(f"F1 Score: {mlp_f1:.4f}")
print(f"Accuracy: {mlp_accuracy:.4f}")
print(f"Weighted Precision: {mlp_precision:.4f}")
print(f"Weighted Recall: {mlp_recall:.4f}")

# Check class distribution
class_dist = test_df.groupBy("label").count().orderBy("count")
class_dist.show(100, truncate=False)

# Calculate baseline accuracy (majority class)
total = test_df.count()
max_class_count = test_df.groupBy("label").count().agg({"count": "max"}).collect()[0][0]
baseline_accuracy = max_class_count / total
print(f"Baseline accuracy (majority class): {baseline_accuracy:.4f}")