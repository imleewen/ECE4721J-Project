![push status](https://focs.ji.sjtu.edu.cn/git/ece472-25su/p1team05/actions/workflows/push.yaml/badge.svg)
![p1m1 release status](https://focs.ji.sjtu.edu.cn/git/ece472-25su/p1team05/actions/workflows/release.yaml/badge.svg?tag=p1m1)
![p1m2 release status](https://focs.ji.sjtu.edu.cn/git/ece472-25su/p1team05/actions/workflows/release.yaml/badge.svg?tag=p1m2)

# RecomMuse

RecomMuse: Where Data Meets Your Playlist

This project is a pure Java implementation of song recommendation system. From the million song dataset, we help you pick the best songs for your playlist.

## Setup

Compile the Maven project

```bash
mvn compile
```

## Milestone 1

### JHDF vs. Alternatives: Why JHDF Wins for Our Use Case

| Feature/Criteria      | JHDF (Winner)            | HDF5-Java (Native)       | SIS-JHDF5 (Pure Java)    | Why JHDF? |
|-----------------------|--------------------------|--------------------------|--------------------------|----------|
| **Pure Java**         |     *Yes** (no native)   |   Requires `libhdf5`     |    Yes                   | No native dependencies for mixed-arch clusters |
| **Read Support**      |    **Optimized for flat**|    Full HDF5             |    Full (slower)         | Faster for flat metadata (artist/tempo/duration) |
| **Write Support**     |    Not needed            |    Yes                   |    Yes                   | **We only read** |
| **Cluster Deployment**|    **Mixed-arch**        |    Arch-specific builds  |    Mixed-arch            | No JNI/native libs to manage |
| **Dependencies**      |    **Single JAR**        |    Native + JNI          |    Pulls Apache SIS      | Zero setup complexity |

To compact several H5 files into an Avro file, run the following command.

```bash
mvn exec:java -Dexec.mainClass="com.p1.Compact" -Dexec.args="<path-to-input-dir> <path-to-output-file>.avro"
```

If you do not know any hierarchical path to the dataset inside the HDF5 file, you can check them easily with the following command.

```bash
h5dump <path-to-H5-file> 
```

To extract information of H5 files inside an Avro file, run the following command.

```bash
mvn exec:java -Dexec.mainClass="com.p1.AvroReader" -Dexec.args="<path-to-avro-file>"
```

## Milestone 2

To be able to implement simple database queries on Drill, the latter should be configured to work in a cluster.

### Prerequisites

1. A running Hadoop cluster with HDFS
2. Apache Drill installed (either embedded or distributed mode)
3. Zookeeper running

### Steps to configure Drill

1. Configure `drill-env.sh` and `drill-override.conf` on all nodes:

    ```sh
    # drill-env.sh on hadoop-master (64GB of RAM available)
    export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}
    export DRILL_HEAP=${DRILL_HEAP:-"8G"}
    export DRILL_MAX_DIRECT_MEMORY=${DRILL_MAX_DIRECT_MEMORY:-"20G"}

    # drill-env.sh on hadoop-slaves
    export JAVA_HOME=${JAVA_HOME:-PATH/TO/JAVA/HOME}
    export DRILL_HEAP=${DRILL_HEAP:-"1G"}
    export DRILL_MAX_DIRECT_MEMORY=${DRILL_MAX_DIRECT_MEMORY:-"2G"}
    ```

    ```yaml
    # drill-override.conf
    drill.exec: {
        cluster-id: "drill-cluster",
        zk.connect: "hadoop-master:2181,hadoop-slave-1:2181,hadoop-slave-2:2181,..."
    }
    ```

2. Start Drill (`drillbit`) service on each node by running `drill-setup.sh` on each node once.

3. Open Drill in the browser

4. Setup the config of the dfs in the Drill browser

    ```json
    {
    "type": "file",
    "connection": "hdfs://hadoop-master:8020",
    "config": {
        "fs.default.name": "hdfs://hadoop-master:8020",
        "hadoop.security.authentication": "simple"
    },
    "workspaces": {
        "msd_avro": {
        "location": "/user/hadoopuser/ms/avro_data",
        "writable": false,
        "defaultInputFormat": "avro",
        "allowAccessOutsideWorkspace": false
        },
        "tmp": {
        "location": "/tmp",
        "writable": true,
        "defaultInputFormat": null,
        "allowAccessOutsideWorkspace": false
        }
    },
    "formats": {
        "avro": {
        "type": "avro",
        "extensions": ["avro"]
        }
    },
    "enabled": true,
    "authMode": "SHARED_USER"
    }
    ```

### Drill Queries

The SQL code of queries can be found in the `simple_queries.sql` file.

### Artist Relationship Analysis

#### Run BFS algorithm on Spark to find shortest paths between artists:

1. Install dependencies

    ```bash
    wget https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.36/slf4j-api-1.7.36.jar
    ```

2. Execute PySpark job:

    ```bash
    spark-submit \
    --conf spark.pyspark.driver.python=/usr/bin/python3 \
    --conf spark.pyspark.python=/usr/bin/python3 \
    --jars /home/hadoopuser/p1team05/sqlite-jdbc-3.50.1.0.jar \
    --conf spark.driver.extraClassPath=/home/hadoopuser/p1team05/sqlite-jdbc-3.50.1.0.jar \
    --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=/home/hadoopuser/spark/conf/log4j2.properties \
    bfs_spark.py AR002UA1187B9A637D AR003FB1187B994355
    ```

    where `AR002UA1187B9A637D` and `AR002UA1187B9A637D` are artist IDs. 

#### Run BFS algorithm on MapReduce to find the shortest paths between artists:

1. Compile the java files with maven to produce `.jar` file needed to run the job on Hadoop MapReduce.

    ```bash
    mvn clean package
    ```

2. In the `run_bfs_experiments.sh` change the variables:

    - `INPUT_PATH`: Tha path to the artist's similarity file in HDFS
    - `BASE_OUTPUT_FILE`: The path to the output folder in HDFS
    - `JAR_PATH`: The path to the JAR file-as a results of compiling java files, you will find a `*-jar-with-dependencies.jar` in the `target` folder
    - `SIMILARITY_DB`: The path to the `artist_similarity.db` database on your local machine

3. Run the `run_bfs_experiments.sh` by giving number of runs needed the following way (here presented for 10 runs):

    ```bash
    ./run_bfs_experiments.sh 10
    ```

### PCA

```bash
python pca.py
```

### Linear Regression

```bash
python linear_regression.py
```

### Linear Regression SGD

```bash
spark-submit   --conf spark.pyspark.driver.python=../pyspark_venv/bin/python3   --conf spark.pyspark.python=../pyspark_venv/bin/python3   --packages org.apache.spark:spark-avro_2.12:3.4.0   sgd_regression_linear.py
```

### Logistic Regression

```bash
spark-submit   --conf spark.pyspark.driver.python=../pyspark_venv/bin/python3   --conf spark.pyspark.python=../pyspark_venv/bin/python3   --packages org.apache.spark:spark-avro_2.12:3.4.0   logistic_regression.py
```

## Year Prediction

### Factorization Machine Regression

- requires package py4j

```bash
conda create -n yp python=3.8
conda activate yp
pip install py4j
python FMR_timbre.py
```

### Linear Regression on timbre

- requires package py4j

```bash
conda activate yp
python linear_timbre.py
```

### MLPClassifier

```bash
python classification.py
```

### Random Forest Classifier

```bash
python random_forest_classification.py
```

## Contributors

Kantaphat Leelakunwet \
Choo Lee Wen \
Ibrahim Daurenov \
Arsen Aghayan
