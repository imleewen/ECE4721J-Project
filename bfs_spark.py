from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, concat_ws
from enum import IntEnum
import time
import sys

'''
spark-submit \
--conf spark.pyspark.driver.python=/usr/bin/python3 \
--conf spark.pyspark.python=/usr/bin/python3 \
--jars /home/hadoopuser/p1team05/sqlite-jdbc-3.50.1.0.jar \
--conf spark.driver.extraClassPath=/home/hadoopuser/p1team05/sqlite-jdbc-3.50.1.0.jar \
--conf spark.driver.extraJavaOptions=-Dlog4j.configuration=/home/hadoopuser/spark/conf/log4j2.properties \
bfs_spark.py
'''

spark = SparkSession.builder.appName("BFS Artist Distance").getOrCreate()

# JDBC URL for SQLite database
jdbc_url = "jdbc:sqlite:/home/hadoopuser/ms/AdditionalFiles/artist_similarity.db"

# Read data from "similarity" table and store in Spark's DataFrame
similarity_df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", "similarity") \
    .load()

# Read data from "artists" table and store in Spark's DataFrame
artists_df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", "artists") \
    .load()

# Left join to include all artists (even those with no similarities)
adj_list_df = artists_df.join(
    similarity_df, 
    artists_df.artist_id == similarity_df.target, 
    "left_outer") \
    .groupBy(artists_df.artist_id) \
    .agg(concat_ws(",", collect_list(similarity_df.similar))
    .alias("similar_artists"))
    
class Status(IntEnum):
    NOT_READY = 0   # Not ready to be vistied
    READY = 1       # Ready to be visited (BFS Frontier)
    VISITED = 2

targetA = sys.argv[1]
targetB = sys.argv[2]

def toNode(row):
    """
    Convert DataFrame row to (target, (similar_artists, distance, status, backpointer)) tuple.
    - `distance`: 0 if target is source (`targetA`), else 10000.
    - `status`: `READY` if target is source, else `NOT_READY`.
    - `backpointer`: None since the current node doesn't know it now.

    Args:
        row (DataFrame): Row with `artist_id` and `similar_artists` columns.

    Returns:
        tuple: (target, (similar_artists, distance, status, backpointer))
    """
    target = row.artist_id
    similar_artists = row.similar_artists.split(",") if row.similar_artists else []
    distance = 0 if target == targetA else 10000
    status = Status.READY if target == targetA else Status.NOT_READY
    backpointer = None
    return (target, (similar_artists, distance, status, backpointer))

# Create RDD for MapReduce
adj_list_rdd = adj_list_df.rdd.map(toNode).cache()

# Define accumulator to count connections to targetB
counter = spark.sparkContext.accumulator(0)


def mapper(node):
    """
    Processes a BFS node, expanding READY nodes and emitting updated neighbors.

    For READY nodes:
        - Generates new entries for each neighbor with updated distance.
        - Tracks hits to targetB via a global counter.
        - Marks the node as VISITED after processing.

    Args:
        node: (target, (similar_artists: list, distance: int, status: Status, backpointer: string))

    Returns:
        list: [
            (newTarget, ([], newDistance, Status.READY, , newBackpointer: string)) for each neighbor,
            (originalTarget, (similar_artists, distance, Status.VISITED, backpointer: string))
        ]
    """
    target, (similar_artists, distance, status, backpointer) = node

    results = []
    
    # If the node is in the BFS frontier, visit it
    if status == Status.READY:
        for artist in similar_artists:
            newTarget = artist
            newDistance = distance + 1
            newStatus = Status.READY
            newBackpointer = target
            if newTarget == targetB:
                counter.add(1)
            
            # New (key, value) pair
            # The [] doesn't really matter, as it will be replaced by the correct similar_artists in reducer.
            newEntry = (newTarget, ([], newDistance, newStatus, newBackpointer))
            results.append(newEntry)
        status = Status.VISITED
    
    # NOTE: Here, there could be some change of status of the current node as the node might become visited.
    #       But instead of changing the status of the current node, we create a new key-value pair and pass
    #       this to reducer to handle the duplicates.
    results.append((target, (similar_artists, distance, status, backpointer)))
    return results


# BFS reduce function
def reducer(data1, data2):
    """
    Reduces two BFS node data entries into one by merging their fields.

    Merging rules:
    - Similar artists: Keeps the non-empty list
    - Distance: Keeps the minimum distance
    - Status: Keeps the most advanced status (VISITED > READY > NOT_READY)
    - Backpointer: Backpointer with shortest distance

    Args:
        data1: (similar_artists: list, distance: int, status: Status, backpointer: string)
        data2: (similar_artists: list, distance: int, status: Status, backpointer: string)

    Returns:
        tuple: Merged (similar_artists, distance, status)
    """
    similar_artists1, distance1, status1, backpointer1 = data1
    similar_artists2, distance2, status2, backpointer2 = data2
    
    # Preserve similar artists
    similar_artists = similar_artists1 if similar_artists1 else similar_artists2
    
    # Preserve minimum distance
    distance = min(distance1, distance2)
    
    # Preserve the most advanced status
    status = max(status1, status2)

    # Preserve the closest backpointer
    backpointer = backpointer1 if distance1 < distance2 else backpointer2
    
    return (similar_artists, distance, status, backpointer)


BFS_ITERATION = 10

start = time.time()

for iteration in range(BFS_ITERATION):
    mapped = adj_list_rdd.flatMap(mapper)
    mapped.collect()  # Trigger computation to update accumulator
    adj_list_rdd = mapped.reduceByKey(reducer).cache()
    
    if counter.value > 0:
        print(f"Target B ({targetB}) was found {iteration + 1} levels from Target A ({targetA}) "
              f"and was connected to {counter.value} artists at that level.")
        
        # Collect all nodes to the driver
        all_nodes = adj_list_rdd.collectAsMap()
        
        # Trace the path back from targetB to targetA
        current = targetB
        path = [current]
        while current != targetA and current is not None:
            # Get the node data
            node_data = all_nodes.get(current, ([], 10000, Status.NOT_READY, None))
            current = node_data[3]  # backpointer
            if current is not None:
                path.append(current)
        
        # Reverse to show path from A to B
        path.reverse()
        
        if path[0] == targetA:
            print("Path found:")
            print(" -> ".join(path))
  
        break

end = time.time()

print("Time taken (ms):", int((end - start) * 1000))

# If no path is found
if counter.value == 0:
    print(f"No path found from Target A ({targetA}) to Target B ({targetB}) within {BFS_ITERATION} iterations.")

# Stop the Spark session
spark.stop()
