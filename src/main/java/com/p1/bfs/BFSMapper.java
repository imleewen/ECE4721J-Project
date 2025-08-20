package com.p1.bfs;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class BFSMapper extends Mapper<Text, Text, Text, Text> {
    
    public static enum Counters {
        TARGETB_CONNECTIONS
    }
    
    private String targetB;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        targetB = conf.get("TARGET_B");
    }

    @Override
    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\\|");
        String similarArtists = parts[0];
        int distance = Integer.parseInt(parts[1]);
        String status = parts[2];
        String backpointer = parts[3];
        
        // Process only nodes in the current frontier
        if ("READY".equals(status)) {
            // Mark current node as visited
            status = "VISITED";
            
            String[] neighbors = similarArtists.split(",");
            if (neighbors.length > 0 && !neighbors[0].isEmpty()) {
                for (String artist : neighbors) {
                    // Emit neighbor with updated distance, status, and backpointer
                    int newDistance = distance + 1;
                    String newValue = "|" + newDistance + "|READY|" + key.toString(); // Empty similar list
                    context.write(new Text(artist), new Text(newValue));
                    
                    // If neighbor is the target, increment counter
                    if (targetB.equals(artist)) {
                        context.getCounter(Counters.TARGETB_CONNECTIONS).increment(1);
                    }
                }
            }
        }
        
        // Emit the original node itself, with its potentially updated status
        String outputValue = similarArtists + "|" + distance + "|" + status + "|" + backpointer;
        context.write(key, new Text(outputValue));
    }
}
