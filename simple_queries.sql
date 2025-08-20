-- The age of the youngest and oldest song
SELECT 
    2025 - MAX(year) AS youngest_song_age,
    2025 - MIN(year) AS oldest_song_age
FROM dfs.root.`/home/hadoopuser/compacted/aggregate.avro`
WHERE year > 0;

-- The hottest song, which is the shortest with the highest energy and lowest tempo
SELECT 
    song_id,
    title
FROM dfs.root.`/home/hadoopuser/compacted/aggregate.avro`
WHERE song_hotttnesss <> 'NaN'
ORDER BY 
    song_hotttnesss DESC,
    duration ASC,
    energy DESC,
    tempo ASC
LIMIT 10;

-- The album with the maximum number of songs in it 
SELECT 
    release,
    COUNT(release) AS ntrack
FROM dfs.root.`/home/hadoopuser/compacted/aggregate.avro`
GROUP BY release
ORDER BY ntrack DESC
LIMIT 1;

-- The artist name of the longest song
SELECT 
    artist_name,
    duration
FROM dfs.root.`/home/hadoopuser/compacted/aggregate.avro`
ORDER BY duration DESC
LIMIT 1;