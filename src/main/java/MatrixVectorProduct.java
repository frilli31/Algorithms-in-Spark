import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;


public class MatrixVectorProduct {

    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]")
                .setAppName("MatrixVectorProduct");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> dataset = sc.textFile("resources/matrix_vector_product.txt").repartition(4);

        // m is the number of columns
        long M = dataset
                .map(x -> x.split(" ")[0])
                .map(Long::valueOf)
                .reduce(Long::max) + 1L;
        long N = dataset
                .map(x -> x.split(" ")[1])
                .map(Long::valueOf)
                .reduce(Long::max) + 1L;
        long sqrt_of_N = (long) Math.sqrt((double) N);

        System.out.println(sqrt_of_N);


        /// Matrix x Vector in R = O(1), M_L = O(N+M), M_A = (N)
        List<Tuple2<Long, Long>> result = dataset
                .mapToPair(row -> {
                    String[] splitted = row.split(" ");
                    return Tuple2.apply(Tuple2.apply(Long.valueOf(splitted[0]), Long.valueOf(splitted[1])), Long.valueOf(splitted[2]));
                })
                .flatMapToPair(t -> {
                    if (t._1._1 == -1L) {
                        return LongStream.range(0, M)
                                .mapToObj(i -> Tuple2.apply(Tuple2.apply(i, t._1._2), t._2))
                                .iterator();
                    } else {
                        return Stream.of(t).iterator();
                    }
                })
                .groupByKey()
                .mapToPair(t -> {
                    long prod = 1;
                    for (long v : t._2)
                        prod = prod * v;
                    return Tuple2.apply(t._1._1, prod);
                })
                .reduceByKey(Long::sum)
                .collect();

        System.out.println("Matrix x Vector in R = O(1), M_L = O(N), M_A = (N): "
                + result.stream()
                .sorted(Comparator.comparing(Tuple2::_1))
                .collect(Collectors.toList())
        );

        /// Matrix x Vector in R = O(1), M_L = O(N^1/2), M_A = (N)
        result = dataset
                .mapToPair(row -> {
                    String[] splitted = row.split(" ");
                    return Tuple2.apply(Tuple2.apply(Long.valueOf(splitted[0]), Long.valueOf(splitted[1])), Long.valueOf(splitted[2]));
                })
                .flatMapToPair(t -> {
                    long i = t._1._1;
                    long j = t._1._2;

                    if (i == -1L) {
                        return LongStream.range(0, M)
                                .mapToObj(z -> Tuple2.apply(Tuple2.apply(z, j % sqrt_of_N), Tuple2.apply(j, t._2)))
                                .iterator();
                    } else {
                        return Stream.of(
                                Tuple2.apply(Tuple2.apply(i, j % sqrt_of_N), Tuple2.apply(j, t._2))
                        ).iterator();
                    }
                })
                .groupByKey()
                .mapToPair(t -> {
                    HashMap<Long, Long> partials = new HashMap<>();
                    for (Tuple2<Long, Long> v : t._2)
                        partials.put(v._1, v._2 * partials.getOrDefault(v._1, 1L));
                    return Tuple2.apply(t._1._1, partials.values().stream().reduce(Long::sum).get());
                })
                .reduceByKey(Long::sum)
                .collect();

        System.out.println("Matrix x Vector in R = O(1), M_L = O(N^1/2), M_A = (N): "
                + result.stream()
                .sorted(Comparator.comparing(Tuple2::_1))
                .collect(Collectors.toList())
        );
    }
}