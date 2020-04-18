import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;


public class SampleSort {

    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]")
                .setAppName("SampleSort");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> dataset = sc.textFile("resources/sort.txt").repartition(4);
        long N = dataset.count();
        long sqrt_of_N = (long) Math.sqrt((double) N);

        long K = sqrt_of_N;
        double threshold = (double) K / N;

        List<Tuple2<Long, Long>> result = dataset
                .map(Long::valueOf)
                .zipWithIndex()
                .flatMapToPair(t -> {
                    Stream<Tuple2<Long, Tuple2<Boolean, Long>>> toReturn = Stream.of(Tuple2.apply(t._2 % sqrt_of_N, Tuple2.apply(false, t._1)));

                    if (Math.random() <= threshold) {
                        Stream<Tuple2<Long, Tuple2<Boolean, Long>>> copies = LongStream
                                .range(0, K)
                                .mapToObj(i -> Tuple2.apply(i, Tuple2.apply(true, t._1)));
                        toReturn = Stream.concat(toReturn, copies);
                    }
                    return toReturn.iterator();
                })
                .groupByKey()
                .flatMapToPair(t -> {
                    ArrayList<Long> regularPairs = new ArrayList<Long>();
                    ArrayList<Long> splitters = new ArrayList<Long>();
                    t._2().forEach(el -> {
                        if (el._1)
                            splitters.add(el._2);
                        else
                            regularPairs.add(el._2);
                    });
                    splitters.sort(Long::compareTo);

                    return regularPairs.stream()
                            .map(el -> Tuple2.apply(splitters.stream().takeWhile(i -> i <= el).count(), el))
                            .iterator();
                })
                .groupByKey()
                .flatMapToPair(t -> {
                    ArrayList<Tuple2<Long, Tuple2<Long, Long>>> toReturn = new ArrayList<>();
                    t._2.forEach(x -> toReturn.add(Tuple2.apply(t._1, Tuple2.apply(x, -1L))));
                    Long size = (long) toReturn.size();

                    toReturn.addAll(
                            LongStream.range(0, 11)
                                    .mapToObj(index -> Tuple2.apply(index, Tuple2.apply(size, t._1)))
                                    .collect(Collectors.toList())
                    );
                    return toReturn.iterator();
                })
                .groupByKey()
                .flatMapToPair(t -> {
                    ArrayList<Long> regularPairs = new ArrayList<Long>();
                    ArrayList<Tuple2<Long, Long>> sizes_of_sets = new ArrayList<>();

                    t._2.forEach(x -> {
                        if (x._2 >= 0)
                            sizes_of_sets.add(x);
                        else
                            regularPairs.add(x._1);
                    });
                    regularPairs.sort(Long::compareTo);
                    long startIndex = sizes_of_sets.stream()
                            .filter(key -> t._1 > key._2)
                            .mapToLong(Tuple2::_1)
                            .sum();

                    return LongStream.range(0, regularPairs.size())
                            .mapToObj(i -> Tuple2.apply(startIndex + i, regularPairs.get((int) i)))
                            .iterator();
                })
                .collect();

        System.out.println("Number of Distinct in R = O(1), M_L = O(N^1/2), M_A = (N): "
                + result.stream()
                .sorted(Comparator.comparing(Tuple2::_1))
                .collect(Collectors.toList())
        );
    }
}
