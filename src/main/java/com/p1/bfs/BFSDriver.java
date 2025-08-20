package com.p1.bfs;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BFSDriver implements Tool {
    private Configuration conf;
    private static String TARGET_A;
    private static String TARGET_B;

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: BFSDriver <input path> <output path> <artist A> <artist B>");
            return -1;
        }

        TARGET_A = args[2];
        TARGET_B = args[3];

        conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", "\t");
        conf.set("TARGET_A", TARGET_A);
        conf.set("TARGET_B", TARGET_B);

        final int MAX_ITERATIONS = 10;
        String inputPath = args[0];
        String basePath = args[1];
        
        // Clean up base output directory
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(new Path(basePath))) {
            fs.delete(new Path(basePath), true);
        }

        // Initialization Job
        System.out.println("Starting Initialization Job");
        Job initJob = Job.getInstance(conf, "BFS Init");
        initJob.setJarByClass(BFSDriver.class);
        initJob.setMapperClass(InitMapper.class);
        initJob.setNumReduceTasks(0); // No reducer needed, mapper does all the work
        initJob.setOutputKeyClass(Text.class);
        initJob.setOutputValueClass(Text.class);
        initJob.setInputFormatClass(KeyValueTextInputFormat.class);
        
        String initOutputPath = basePath + "/iter0";
        FileInputFormat.addInputPath(initJob, new Path(inputPath));
        FileOutputFormat.setOutputPath(initJob, new Path(initOutputPath));
        
        if (!initJob.waitForCompletion(true)) {
            System.err.println("Initialization job failed!");
            return 1;
        }
        
        String currentInputPath = initOutputPath;

        // BFS Iteration Jobs
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            System.out.println("Starting BFS Iteration " + (i + 1));
            Job job = Job.getInstance(conf, "BFS Iter " + (i + 1));
            job.setJarByClass(BFSDriver.class);
            job.setMapperClass(BFSMapper.class);
            job.setReducerClass(BFSReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            job.setInputFormatClass(KeyValueTextInputFormat.class);
            
            String currentOutputPath = basePath + "/iter" + (i + 1);
            FileInputFormat.addInputPath(job, new Path(currentInputPath));
            FileOutputFormat.setOutputPath(job, new Path(currentOutputPath));
            
            if (!job.waitForCompletion(true)) {
                System.err.println("BFS iteration " + (i + 1) + " failed!");
                return 1;
            }
            
            currentInputPath = currentOutputPath;
            
            long connections = job.getCounters()
                                  .findCounter(BFSMapper.Counters.TARGETB_CONNECTIONS)
                                  .getValue();

            if (connections > 0) {
                System.out.printf("Target B (%s) was found %d levels from Target A (%s)\n", TARGET_B, (i + 1), TARGET_A);
                System.out.printf("It was connected to %d artists at that level.\n", connections);
                tracePath(currentInputPath); // Call path tracing
                return 0; // Success
            }
        }
        
        System.out.printf("No path found from Target A to Target B within %d iterations.\n", MAX_ITERATIONS);
        return 0;
    }

    private void tracePath(String finalOutputPath) throws Exception {
        System.out.println("Tracing path from B to A...");
        Map<String, String> nodeDataMap = new HashMap<>();

        // Read the final output file(s) from HDFS and load into a map
        FileSystem fs = FileSystem.get(conf);
        Path outputPath = new Path(finalOutputPath);
        if (fs.exists(outputPath)) {
            // Assuming single reducer output file part-r-00000
            Path resultFile = new Path(finalOutputPath + "/part-r-00000");
            if (fs.exists(resultFile)) {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(resultFile)))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\t", 2);
                        if (parts.length == 2) {
                            nodeDataMap.put(parts[0], parts[1]);
                        }
                    }
                }
            }
        }
        
        if (nodeDataMap.isEmpty() || !nodeDataMap.containsKey(TARGET_B)) {
            System.out.println("Could not find Target B in the final output to trace path.");
            return;
        }

        // Reconstruct the path by following backpointers
        List<String> path = new ArrayList<>();
        String current = TARGET_B;
        while (current != null && !current.equals("null") && !current.equals(TARGET_A)) {
            path.add(current);
            String data = nodeDataMap.get(current);
            if (data == null) {
                System.err.println("Path broken. Node not found: " + current);
                return;
            }
            String[] parts = data.split("\\|");
            current = (parts.length > 3) ? parts[3] : null;
        }

        if (current != null && current.equals(TARGET_A)) {
            path.add(TARGET_A);
            Collections.reverse(path);
            System.out.println("Path found:");
            System.out.println(String.join(" -> ", path));
        } else {
            System.out.println("A complete path back to Target A could not be traced.");
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("--- Starting BFSDriver ---");
        System.out.println("Number of arguments received: " + args.length);
        for (int i = 0; i < args.length; i++) {
            System.out.println("Argument [" + i + "]: " + args[i]);
        }
        System.out.println("--------------------------");

        int res = ToolRunner.run(new Configuration(), new BFSDriver(), args);
        System.exit(res);
    }
}
