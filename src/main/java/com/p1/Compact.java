package com.p1;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.avro.file.CodecFactory;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.specific.SpecificDatumWriter;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;

public class Compact {
    private final File inputDir; // Input directory of H5 files to be compacted
    private final File outputFile; // Output avro file

    // Constructor to set up variables
    public Compact(String inputDirPath, String outputFilePath) {
        this.inputDir = new File(inputDirPath);
        this.outputFile = new File(outputFilePath);
    }

    public void compactFiles() throws IOException {
        SpecificDatumWriter<H5Fields> datumWriter = new SpecificDatumWriter<>(H5Fields.class);

        try (DataFileWriter <H5Fields> writer = new DataFileWriter<>(datumWriter)) {
            // Enable Snappy compresion
            writer.setCodec(CodecFactory.snappyCodec());
            writer.create(H5Fields.getClassSchema(), outputFile);

            // Process all .h5 files recursively
            Files.walk(inputDir.toPath())
                .filter(p -> p.toString().endsWith(".h5"))
                .forEach(p -> {
                    try {
                        // Create a record to write
                        H5Fields record = new H5Fields();
                        
                        // Extract information from H5 files
                        extract(record, p.toString());
                        
                        // Write record to avro file
                        writer.append(record);
                    } catch (IOException e) {
                        System.err.println("Error processing " + p + ": " + e.getMessage());
                    }
                });
        }
    }

    /**
     * Extract information from H5 files and write it to H5Fields record defined in Avro.
     * 
     * NOTE: If there is any changes to the variables needed for data analysis,
     *       need to make modification here as well. The dataset paths that need
     *       to be traversed are predefined. Values of each field that will be set
     *       to H5Fields record are also needed to manually changed.
     * 
     *       For LinkedHashMap, every value is an array of size 1, hence Array.get(value, 0)
     *       is used to get the first element.
     * 
     *       For Object[], this is likely an empty or null value. It is handled
     *       separately by still passing null value to the record.
     */
    private static void extract(H5Fields record, String filename) throws IOException {
        try (HdfFile hdfFile = new HdfFile(Paths.get(filename))) {

            String[] datasetPaths = {
                "/analysis/bars_confidence",
                "/analysis/beats_confidence",
                "/analysis/sections_confidence",
                "/analysis/segments_confidence",
                "/analysis/segments_loudness_max",
                "/analysis/segments_pitches",
                "/analysis/segments_timbre",
                "/analysis/tatums_confidence",
                "/analysis/songs", 
                "/metadata/songs",
                "/musicbrainz/songs"
            };

            for (String datasetPath : datasetPaths) {
                // Get Dataset
                Dataset dataset = hdfFile.getDatasetByPath(datasetPath);
                
                // Get data of the Dataset
                Object data = dataset.getDataFlat();    

                // Determine the type of data of process accordingly
                if (data instanceof LinkedHashMap) {
                    LinkedHashMap<String, Object> mapData = (LinkedHashMap<String, Object>) data;

                    switch (datasetPath) {
                        case "/analysis/songs":
                            record.setAnalysisSampleRate((Integer) Array.get(mapData.get("analysis_sample_rate"), 0));
                            record.setDanceability((Double) Array.get(mapData.get("danceability"), 0));
                            record.setDuration((Double) Array.get(mapData.get("duration"), 0));
                            record.setEndOfFadeIn((Double) Array.get(mapData.get("end_of_fade_in"), 0));
                            record.setEnergy((Double) Array.get(mapData.get("energy"), 0));
                            record.setKey((Integer) Array.get(mapData.get("key"), 0));
                            record.setKeyConfidence((Double) Array.get(mapData.get("key_confidence"), 0));
                            record.setLoudness((Double) Array.get(mapData.get("loudness"), 0));
                            record.setMode((Integer) Array.get(mapData.get("mode"), 0));
                            record.setModeConfidence((Double) Array.get(mapData.get("mode_confidence"), 0));
                            record.setStartOfFadeOut((Double) Array.get(mapData.get("start_of_fade_out"), 0));
                            record.setTempo((Double) Array.get(mapData.get("tempo"), 0));
                            record.setTimeSignature((Integer) Array.get(mapData.get("time_signature"), 0));
                            record.setTimeSignatureConfidence((Double) Array.get(mapData.get("time_signature_confidence"), 0));
                            break;
                        case "/metadata/songs":
                            record.setArtist7digitalid((Integer) Array.get(mapData.get("artist_7digitalid"), 0));
                            record.setArtistFamiliarity((Double) Array.get(mapData.get("artist_familiarity"), 0));
                            record.setArtistHotttnesss((Double) Array.get(mapData.get("artist_hotttnesss"), 0));
                            record.setArtistLatitude((Double) Array.get(mapData.get("artist_latitude"), 0));
                            record.setArtistLongitude((Double) Array.get(mapData.get("artist_longitude"), 0));
                            record.setArtistPlaymeid((Integer) Array.get(mapData.get("artist_playmeid"), 0));
                            record.setRelease7digitalid((Integer) Array.get(mapData.get("release_7digitalid"), 0));
                            record.setSongHotttnesss((Double) Array.get(mapData.get("song_hotttnesss"), 0));
                            record.setTrack7digitalid((Integer) Array.get(mapData.get("track_7digitalid"), 0));
                            break;
                        case "/musicbrainz/songs":
                            record.setYear((Integer) Array.get(mapData.get("year"), 0));
                            break;
                        default:
                            break;
                    }
                    
                } else if (data instanceof double[]) {
                    double[] primitiveArray = (double[]) data;

                    double mean = Arrays.stream(primitiveArray).average().orElse(0.0);
                    double max = Arrays.stream(primitiveArray).max().orElse(0.0);
                    double std = Math.sqrt(Arrays.stream(primitiveArray)
                                            .map(v -> Math.pow(v - mean, 2))
                                            .average().orElse(0.0));

                    switch (datasetPath) {
                        case "/analysis/bars_confidence":
                            record.setBarsConfidenceMean((Double) mean);
                            record.setBarsConfidenceMax((Double) max);
                            record.setBarsConfidenceStd((Double) std);
                            break;
                        case "/analysis/beats_confidence":
                            record.setBeatsConfidenceMean((Double) mean);
                            record.setBeatsConfidenceMax((Double) max);
                            record.setBeatsConfidenceStd((Double) std);
                            break;
                        case "/analysis/sections_confidence":
                            record.setSectionsConfidenceMean((Double) mean);
                            record.setSectionsConfidenceMax((Double) max);
                            record.setSectionsConfidenceStd((Double) std);
                            break;
                        case "/analysis/segments_confidence":
                            record.setSegmentsConfidenceMean((Double) mean);
                            record.setSegmentsConfidenceMax((Double) max);
                            record.setSegmentsConfidenceStd((Double) std);
                            break;
                        case "/analysis/segments_loudness_max":
                            record.setSegmentsLoudnessMaxMean((Double) mean);
                            record.setSegmentsLoudnessMaxStd((Double) std);
                            break;
                        case "/analysis/tatums_confidence":
                            record.setTatumsConfidenceMean((Double) mean);
                            record.setTatumsConfidenceMax((Double) max);
                            record.setTatumsConfidenceStd((Double) std);
                            break;
                        case "/analysis/segments_pitches":
                            int n = primitiveArray.length / 12;
                            double[] means = calculateMeans(primitiveArray, n);
                            double[] covarianceArray = calculateCovarianceArray(primitiveArray, n, means);
                            double[] combinedArray = new double[means.length + covarianceArray.length];
                            System.arraycopy(means, 0, combinedArray, 0, means.length);
                            System.arraycopy(covarianceArray, 0, combinedArray, means.length, covarianceArray.length);

                            List<Double> doubleList = Arrays.stream(combinedArray)
                                                .boxed()
                                                .collect(Collectors.toList());
                            record.setSegmentsPitches(doubleList);
                            break;
                        case "/analysis/segments_timbre":
                            n = primitiveArray.length / 12;
                            means = calculateMeans(primitiveArray, n);
                            covarianceArray = calculateCovarianceArray(primitiveArray, n, means);
                            combinedArray = new double[means.length + covarianceArray.length];
                            System.arraycopy(means, 0, combinedArray, 0, means.length);
                            System.arraycopy(covarianceArray, 0, combinedArray, means.length, covarianceArray.length);

                            doubleList = Arrays.stream(combinedArray)
                                                .boxed()
                                                .collect(Collectors.toList());
                            record.setSegmentsTimbre(doubleList);
                            break;
                        default:
                            break;
                    }

                } else if (data instanceof Object[]) {
                    // This is for handling NULL data.
                    switch (datasetPath) {
                        case "/analysis/bars_confidence":
                            record.setBarsConfidenceMean(null);
                            record.setBarsConfidenceMax(null);
                            record.setBarsConfidenceStd(null);
                            break;
                        case "/analysis/beats_confidence":
                            record.setBeatsConfidenceMean(null);
                            record.setBeatsConfidenceMax(null);
                            record.setBeatsConfidenceStd(null);
                            break;
                        case "/analysis/sections_confidence":
                            record.setSectionsConfidenceMean(null);
                            record.setSectionsConfidenceMax(null);
                            record.setSectionsConfidenceStd(null);
                            break;
                        case "/analysis/segments_loudness_max":
                            record.setSegmentsLoudnessMaxMean(null);
                            record.setSegmentsLoudnessMaxStd(null);
                            break;
                        case "/analysis/tatums_confidence":
                            record.setTatumsConfidenceMean(null);
                            record.setTatumsConfidenceMax(null);
                            record.setTatumsConfidenceStd(null);
                            break;
                        case "/analysis/segments_pitches":
                            record.setSegmentsPitches(
                                Arrays.stream((Object[]) data)
                                    .map(obj -> obj instanceof Double ? (Double) obj : null)
                                    .collect(Collectors.toList())
                            );
                            break;
                        case "/analysis/segments_timbre":
                            record.setSegmentsTimbre(
                                Arrays.stream((Object[]) data)
                                    .map(obj -> obj instanceof Double ? (Double) obj : null)
                                    .collect(Collectors.toList())
                            );
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }

    private static double[] calculateMeans(double[] flatArray, int n) {
        double[] means = new double[12];
        
        for (int col = 0; col < 12; col++) {
            double sum = 0.0;
            for (int row = 0; row < n; row++) {
                sum += flatArray[row * 12 + col];
            }
            means[col] = sum / n;
        }
        
        return means;
    }

    private static double[] calculateCovarianceArray(double[] flatArray, int n, double[] means) {
        double[] covArray = new double[78];

        int k = 0;
        
        for (int i = 0; i < 12; i++) {
            for (int j = i; j < 12; j++) {
                double covariance = 0.0;
                
                for (int row = 0; row < n; row++) {
                    double diffI = flatArray[row * 12 + i] - means[i];
                    double diffJ = flatArray[row * 12 + j] - means[j];
                    covariance += diffI * diffJ;
                }
                
                covariance /= (n - 1); // sample covariance
                covArray[k++] = covariance;
            }
        }
        
        if (k == 78)
            return covArray;
        else
            throw new IllegalStateException("Covariance calculation failed: expected 78 values but got " + k);
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: Compact <input_dir> <output_file>");
            System.exit(0);
        }

        try {
            Compact compactor = new Compact(args[0], args[1]);
            compactor.compactFiles();
            System.out.println("H5 files compacted successfully into " + args[1]);
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}
