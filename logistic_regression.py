from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Start Spark session
spark = SparkSession.builder \
    .appName("Multiclass Logistic Regression") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .getOrCreate()

# Load AVRO data
df = spark.read.format("avro").load("file:///home/hadoopuser/p1team05/combined.avro")

# Flatten pitch and timbre vectors into columns
for i in range(12):
    df = (df
        .withColumn(f"pitch_mean_{i}",  F.col("segments_pitches_mean")[i])
        .withColumn(f"pitch_std_{i}",   F.col("segments_pitches_std")[i])
        .withColumn(f"timbre_mean_{i}", F.col("segments_timbre_mean")[i])
        .withColumn(f"timbre_std_{i}",  F.col("segments_timbre_std")[i])
    )

# Select numeric features
numeric_cols = [f.name for f in df.schema.fields
    if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType)) and f.name != "year"
]

# Drop nulls and define class labels (decade buckets)
df = df.na.drop(subset=numeric_cols + ["year"]).filter(F.col("year") != 0)

# Define feature pipeline: Assemble -> Scale -> PCA
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
scaler    = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
pca       = PCA(k=32, inputCol="scaledFeatures", outputCol="pcaFeatures")

pipeline = Pipeline(stages=[assembler, scaler, pca])
model = pipeline.fit(df)
df_trans = model.transform(df)

# Train/test split
train_df, test_df = df_trans.randomSplit([0.8, 0.2], seed=42)

# Train multiclass logistic regression
lr = LogisticRegression(
    featuresCol="pcaFeatures",
    labelCol="year",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

lr_model = lr.fit(train_df)

# Evaluate on test data
predictions = lr_model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="year", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

for metric in ["f1", "weightedPrecision", "weightedRecall"]:
    evaluator.setMetricName(metric)
    print(f"{metric}: {evaluator.evaluate(predictions):.4f}")

conf_matrix = predictions.groupBy("year", "prediction").count().orderBy("year", "prediction")
conf_matrix.show()

spark.stop()

