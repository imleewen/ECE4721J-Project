from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline
import numpy as np

# Config
THRESHOLD = 0.90          # Target cumulative explained variance
MIN_STD = 1e-8            # Minimum std threshold to filter out near-constant features
MAX_K = 100               # Max number of components to try in PCA

# Start Spark session
spark = SparkSession.builder \
    .appName("PCA_FIX") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configurationFile=/home/hadoopuser/spark/conf/log4j2.properties") \
    .getOrCreate()

# Load dataset (AVRO format)
df = spark.read.format("avro").load("file:///home/hadoopuser/p1team05/combined.avro")

# Flatten 12-dim arrays: pitches and timbre (mean + std)
for i in range(12):
    df = df.withColumn(f"pitch_mean_{i}",  F.col("segments_pitches_mean")[i])
    df = df.withColumn(f"pitch_std_{i}",   F.col("segments_pitches_std")[i])
    df = df.withColumn(f"timbre_mean_{i}", F.col("segments_timbre_mean")[i])
    df = df.withColumn(f"timbre_std_{i}",  F.col("segments_timbre_std")[i])

# Select numeric columns (exclude non-useful ones)
raw_numeric = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
    and f.name != "year"
]

# Remove irrelevant features (e.g., IDs, sample_rate, etc.)
bad_tokens = ("id", "playmeid", "sample_rate", "7digital")
numeric_cols = [c for c in raw_numeric if not any(tok in c.lower() for tok in bad_tokens)]

print("=" * 60)
#print("Initial numeric cols:", len(raw_numeric))
print("Numeric cols:", numeric_cols)
print("=" * 60)

# Drop nulls and rows with year == 0
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

# PCA Setup
k = min(MAX_K, len(numeric_cols))
pca = PCA(k=k, inputCol="scaledFeatures", outputCol="pcaFeatures")

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
    n = int(np.argmax(mask) + 1)
    print(f"\n Components needed to reach >= {int(THRESHOLD*100)}% variance: {n}")
else:
    print(f"\n Could not reach {int(THRESHOLD*100)}% variance. Total explained: {cum[-1]*100:.2f}% "
          f"using {len(explained)} components. Consider cleaning the data further.")



from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F
import pandas as pd
import os

#Correlation of the scaled raw features
scaled_df = model.transform(df).select("scaledFeatures")  # Vector column

corr_matrix = Correlation.corr(scaled_df, "scaledFeatures", "pearson").head()[0]
# Convert to NumPy -> Pandas for readability
corr_arr = corr_matrix.toArray()
corr_pd  = pd.DataFrame(corr_arr, index=numeric_cols, columns=numeric_cols)

#save to csv
corr_pd.to_csv("features_correlation_table.csv")

print("\nPearson correlation of raw *scaled* features ")
print(corr_pd.round(2))


spark.stop()

