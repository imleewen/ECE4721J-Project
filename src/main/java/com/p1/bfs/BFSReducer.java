package com.p1.bfs;

import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class BFSReducer extends Reducer<Text, Text, Text, Text> {
    
    @Override
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        String similarArtists = "";
        int minDistance = 10000;
        String status = "NOT_READY";
        String bestBackpointer = "null";
        
        for (Text value : values) {
            String[] parts = value.toString().split("\\|");
            
            // Part 0: Similar Artists (adjacency list)
            // Preserve the actual list, which only comes from the original node record
            if (parts[0] != null && !parts[0].isEmpty()) {
                similarArtists = parts[0];
            }
            
            // Part 1: Distance
            int currentDistance = Integer.parseInt(parts[1]);
            
            // Part 2: Status
            String currentStatus = parts[2];

            // Part 3: Backpointer
            String currentBackpointer = parts[3];

            // If we've found a shorter path, update distance and backpointer
            if (currentDistance < minDistance) {
                minDistance = currentDistance;
                bestBackpointer = currentBackpointer;
            }
            
            // Update status based on precedence: VISITED > READY > NOT_READY
            if (currentStatus.equals("VISITED")) {
                status = "VISITED";
            } else if (currentStatus.equals("READY") && !status.equals("VISITED")) {
                status = "READY";
            }
        }
        
        String outputValue = similarArtists + "|" + minDistance + "|" + status + "|" + bestBackpointer;
        context.write(key, new Text(outputValue));
    }
}
