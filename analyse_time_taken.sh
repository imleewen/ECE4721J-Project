#!/bin/bash

# Check if log file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <log_file>"
    exit 1
fi

LOG_FILE=$1

# Check if file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: File $LOG_FILE not found!"
    exit 1
fi

echo "Processing log file: $LOG_FILE"
echo ""

# Process the file and collect all data
results=$(awk '
BEGIN {
    current_run = 0
    sum_map_time = 0
    sum_reduce_time = 0
    in_timing_section = 0
    total_map = 0
    total_reduce = 0
    count = 0
}
/Run [0-9]+ - Artist A:/ {
    if (current_run > 0) {
        printf "%d %d %d\n", current_run, sum_map_time, sum_reduce_time
        total_map += sum_map_time
        total_reduce += sum_reduce_time
        count++
    }
    current_run = $2
    sum_map_time = 0
    sum_reduce_time = 0
    in_timing_section = 0
}
/===== Timing Information =====/ {
    in_timing_section = 1
}
/Total time spent by all map tasks \(ms\)=/ && in_timing_section {
    split($0, parts, "=")
    sum_map_time += parts[2]
}
/Total time spent by all reduce tasks \(ms\)=/ && in_timing_section {
    split($0, parts, "=")
    sum_reduce_time += parts[2]
}
/=============================/ {
    in_timing_section = 0
}
END {
    if (current_run > 0) {
        printf "%d %d %d\n", current_run, sum_map_time, sum_reduce_time
        total_map += sum_map_time
        total_reduce += sum_reduce_time
        count++
    }
    printf "TOTAL %d %d %d\n", total_map, total_reduce, count
}
' "$LOG_FILE")

# Display per-run results
echo "Individual Run Statistics:"
echo "--------------------------------------------------"
printf "%-6s %-23s %-20s\n" "Run" "Mapper Time (ms)" "Reducer Time (ms)"
echo "--------------------------------------------------"
echo "$results" | grep -v "^TOTAL" | while read run map reduce; do
    printf "%-6d %-23d %-20d\n" "$run" "$map" "$reduce"
done
echo "--------------------------------------------------"

# Display aggregate statistics
total_info=$(echo "$results" | grep "^TOTAL")
total_map=$(echo "$total_info" | awk '{print $2}')
total_reduce=$(echo "$total_info" | awk '{print $3}')
count=$(echo "$total_info" | awk '{print $4}')

if [ "$count" -gt 0 ]; then
    avg_map=$((total_map / count))
    avg_reduce=$((total_reduce / count))
    
    echo ""
    echo "Aggregate Statistics:"
    echo "--------------------------------------------------"
    printf "%-30s %-20d\n" "Total Mapper Time (ms):" "$total_map"
    printf "%-30s %-20d\n" "Total Reducer Time (ms):" "$total_reduce"
    printf "%-30s %-20d\n" "Combined Total Time (ms):" "$((total_map + total_reduce))"
    echo "--------------------------------------------------"
    printf "%-30s %-20d\n" "Average Mapper Time (ms):" "$avg_map"
    printf "%-30s %-20d\n" "Average Reducer Time (ms):" "$avg_reduce"
    printf "%-30s %-20d\n" "Average Combined Time (ms):" "$(( (total_map + total_reduce) / count ))"
    echo "--------------------------------------------------"
    printf "%-30s %-20d\n" "Number of Runs:" "$count"
    echo "--------------------------------------------------"
fi
