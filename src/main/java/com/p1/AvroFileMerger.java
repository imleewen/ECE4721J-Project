package com.p1;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileReader;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericDatumWriter;
import org.apache.avro.generic.GenericRecord;

public class AvroFileMerger {

    public static void mergeAvroFiles(List<String> inputFilePaths, String outputFilePath) throws IOException {
        if (inputFilePaths == null || inputFilePaths.isEmpty()) {
            throw new IllegalArgumentException("Input file list cannot be empty");
        }

        // Initialize variables
        Schema schema = null;
        DataFileWriter<GenericRecord> writer = null;
        GenericDatumReader<GenericRecord> datumReader = new GenericDatumReader<>();

        try {
            // First pass to get the schema from the first file
            try (DataFileReader<GenericRecord> firstFileReader = new DataFileReader<>(
                    new File(inputFilePaths.get(0)), datumReader)) {
                schema = firstFileReader.getSchema();
            }

            // Create writer with the schema
            writer = new DataFileWriter<>(new GenericDatumWriter<>(schema));
            writer.create(schema, new File(outputFilePath));

            // Process all input files
            for (String inputPath : inputFilePaths) {
                try (DataFileReader<GenericRecord> reader = new DataFileReader<>(
                        new File(inputPath), datumReader)) {
                    
                    // Verify schema matches
                    if (!reader.getSchema().equals(schema)) {
                        throw new IOException("Schema mismatch in file: " + inputPath);
                    }

                    // Copy all records from this file to the output
                    for (GenericRecord record : reader) {
                        writer.append(record);
                    }
                }
                System.out.println("Merged " + inputPath);
            }
        } finally {
            if (writer != null) {
                writer.close();
            }
        }
    }

    @SuppressWarnings("CallToPrintStackTrace")
    public static void main(String[] args) {
        try {
            List<String> inputFiles = new ArrayList<>();
            for (char c = 'A'; c <= 'Z'; c++) {
                inputFiles.add("./output_" + c + ".avro");
            }
            mergeAvroFiles(inputFiles, "output_A2Z.avro");
            System.out.println("Avro files merged successfully!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
