from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline

# mllib (RDD) imports
from pyspark.mllib.regression import LinearRegressionWithSGD, LabeledPoint
from pyspark.mllib.linalg import Vectors as OldVectors
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.evaluation import RegressionEvaluator


# Spark Session
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
# Normalize the label 'year'
year_stats = df.select(F.mean("year").alias("mean"), F.stddev("year").alias("std")).first()
year_mean = year_stats["mean"]
year_std = year_stats["std"]

df = df.withColumn("year", (F.col("year") - year_mean) / year_std)


df.select("year").summary().show()

# Assemble features
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

# Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)

pca = PCA(k=32, inputCol="scaledFeatures", outputCol="pcaFeatures")
pipeline = Pipeline(stages=[assembler, scaler, pca])

model = pipeline.fit(df)
df_trans  = model.transform(df)
print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaAAAAAAAAAAAAAAAAAAAAAAAA')
print("Mean year:", df.select(F.mean("year")).first())
print("Min year:", df.select(F.min("year")).first())
print("Max year:", df.select(F.max("year")).first())
print("Sample features + label:")
df_trans.select("scaledFeatures", "year").show(5, truncate=False)


# Train / test RDD split
train_df, test_df = df_trans.randomSplit([0.8, 0.2], seed=42)


def row_to_lp(row):
    # Convert MLlib Vector -> old mllib dense vector
    feats = OldVectors.dense(row['pcaFeatures'].toArray())
    label = float(row['year'])
    return LabeledPoint(label, feats)

train_rdd = train_df.rdd.map(row_to_lp).cache()
test_rdd  = test_df.rdd.map(row_to_lp).cache()


# SGD regressor
sgd_model = LinearRegressionWithSGD.train(
    data            = train_rdd,
    iterations      = 800,          
    step            = 0.2,
    regParam = 0.001,
    miniBatchFraction = 0.1
            
)

# Evaluation
predictions = test_rdd.map(lambda lp: (float(sgd_model.predict(lp.features)), lp.label))
metrics = RegressionMetrics(predictions)
print(f"RMSE: {metrics.rootMeanSquaredError}")
print(f"MSE: {metrics.meanSquaredError}")
print(f"R^2: {metrics.r2}")
print(f"MAE: {metrics.meanAbsoluteError}")


denorm_preds = predictions.map(lambda x: (
    x[0] * year_std + year_mean,
    x[1] * year_std + year_mean
)).toDF(["prediction", "label"])
denorm_preds.show(10)
print("Corr:", denorm_preds.stat.corr("prediction", "label"))



spark.stop()

