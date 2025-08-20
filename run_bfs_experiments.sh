#!/bin/bash

# Paths
INPUT_PATH="/user/hadoopuser/bfs_input/output.txt"
BASE_OUTPUT_PATH="/user/hadoopuser/bfs_output"
JAR_PATH="/home/hadoopuser/Documents/p1team05/target/p1team05-1.0-SNAPSHOT-jar-with-dependencies.jar" 
SIMILARITY_DB="/home/hadoopuser/ms/AdditionalFiles/artist_similarity.db"

# Number of runs
NUM_RUNS=$1

# Get total number of artists
TOTAL_ARTISTS=$(sqlite3 "$SIMILARITY_DB" "SELECT COUNT(*) FROM artists;")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="bfs_timings_$TIMESTAMP.log"

echo "BFS Performance Test - $TIMESTAMP" > $LOG_FILE
echo "=================================" >> $LOG_FILE

for ((i=1; i<=$NUM_RUNS; i++)); do
    echo "Running BFS iteration $i of $NUM_RUNS"
    
    # Generate two random numbers between 1 and TOTAL_ARTISTS
    RAND1=$((1 + RANDOM % TOTAL_ARTISTS))
    RAND2=$((1 + RANDOM % TOTAL_ARTISTS))
    
    # Ensure they're different
    while [ "$RAND1" -eq "$RAND2" ]; do
        RAND2=$((1 + RANDOM % TOTAL_ARTISTS))
    done
    
    # Get artist IDs
    ARTIST_A=$(sqlite3 "$SIMILARITY_DB" "SELECT artist_id FROM artists LIMIT 1 OFFSET $((RAND1-1));")
    ARTIST_B=$(sqlite3 "$SIMILARITY_DB" "SELECT artist_id FROM artists LIMIT 1 OFFSET $((RAND2-1));")
    
    echo "Artist A: $ARTIST_A"
    echo "Artist B: $ARTIST_B"

    echo "Run $i - Artist A: $ARTIST_A, Artist B: $ARTIST_B" >> $LOG_FILE
    
    # Create output directory for this run
    RUN_OUTPUT="$BASE_OUTPUT_PATH/run_$i"
    mkdir -p "$RUN_OUTPUT"
    
    echo "Starting Hadoop job..."
    # Run Hadoop job
    hadoop jar "$JAR_PATH" \
        "$INPUT_PATH" \
        "$RUN_OUTPUT" \
        "$ARTIST_A" \
        "$ARTIST_B" 2>&1 | tee -a temp.log

    echo "===== Timing Information =====" >> $LOG_FILE
    grep -E "Total time spent by all map|Total time spent by all reduce" temp.log >> $LOG_FILE
    echo "=============================" >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    # Clean up
    rm temp.log

    
    echo "Completed iteration $i"
    echo "----------------------------------------"
done

echo "All $NUM_RUNS BFS runs completed!"

./analyse_time_taken.sh $LOG_FILE > "analysis.log"
