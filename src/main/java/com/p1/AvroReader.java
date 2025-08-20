package com.p1;

import java.io.IOException;
import java.nio.file.Paths;

import org.apache.avro.file.DataFileReader;
import org.apache.avro.specific.SpecificDatumReader;

public class AvroReader {
    public static void main(String[] args) throws IOException {
        int count = 0;

        if (args.length != 1) {
            System.out.println("Usage: AvroReader <input_avro_file>");
            System.exit(0);
        }

        // Create Avro reader using the embedded schema
        SpecificDatumReader < H5Fields > datumReader = new SpecificDatumReader < > (H5Fields.class);

        try (DataFileReader < H5Fields > avroReader = new DataFileReader < > (
            Paths.get(args[0]).toFile(),
            datumReader
        );) {

            // Process each record
            while (avroReader.hasNext()) {
                H5Fields record = avroReader.next();

                // Extract H5 data from Avro record
                // System.out.printf("%-30s: %s%n", "tempo", record.getTempo());
                // System.out.printf("%-30s: %s%n", "danceability", record.getDanceability());
                // System.out.printf("%-30s: %s%n", "energy", record.getEnergy());
                // System.out.printf("%-30s: %s%n", "duration", record.getDuration());
                // System.out.printf("%-30s: %s%n", "bars_confidence_max", record.getBarsConfidenceMax());
                // System.out.printf("%-30s: %s%n", "bars_confidence_mean", record.getBarsConfidenceMean());
                // System.out.printf("%-30s: %s%n", "bars_confidence_std", record.getBarsConfidenceStd());
                // System.out.printf("%-30s: %s%n", "segments_pitches_mean.size", record.getSegmentsPitchesMean().size());
                // System.out.printf("%-30s: %s%n", "segments_pitches_mean", record.getSegmentsPitchesMean());
                // System.out.printf("%-30s: %s%n", "segments_pitches_std.size", record.getSegmentsPitchesStd().size());
                // System.out.printf("%-30s: %s%n", "segments_pitches_std", record.getSegmentsPitchesStd());
                System.out.printf("%-30s: %s%n", "segments_pitches", record.getSegmentsPitches());
                System.out.printf("%-30s: %s%n", "segments_pitches.size", record.getSegmentsPitches().size());
                System.out.printf("%-30s: %s%n", "segments_timbre", record.getSegmentsTimbre());
                System.out.printf("%-30s: %s%n", "segments_timbre.size", record.getSegmentsTimbre().size());
                count++;
            }
        }

        System.out.println(count);
    }
}
