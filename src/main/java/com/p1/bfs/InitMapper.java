package com.p1.bfs;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class InitMapper extends Mapper<Text, Text, Text, Text> {
    private String targetA;
    private static final int INFINITY = 10000;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        targetA = conf.get("TARGET_A");
    }
    
    @Override
    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String artistId = key.toString();
        String similarArtists = value.toString();
        
        String status;
        int distance;
        
        if (artistId.equals(targetA)) {
            status = "READY";
            distance = 0;
        } else {
            status = "NOT_READY";
            distance = INFINITY;
        }
        
        // Output format: similar_artists|distance|status|backpointer
        String outputValue = similarArtists + "|" + distance + "|" + status + "|null";
        context.write(key, new Text(outputValue));
    }
}
