import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.LongStream;
import java.util.stream.Stream;


public class MaxPairwiseDistance {

    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true)
                .setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> dataset = sc.textFile("resources/max_pairwise_distance.txt").repartition(4);
        Long N = dataset.count();
        Long sqrt_of_N = (long) Math.sqrt((double) N);

        Long max_length;

        /// MAXIMUM PAIRWISE DISTANCE in R = O(1), M_L = O(N^1/2), M_A = (N^2)

        max_length = dataset
                .zipWithIndex()
                .flatMapToPair(row -> {  // ROUND 1 : Map
                    return LongStream
                            .range(0, sqrt_of_N)
                            .mapToObj(j -> new Tuple2<>(row._2, j))
                            .map(key -> new Tuple2<>(key, Long.valueOf(row._1())))
                            .iterator();
                })
                .flatMapToPair((tuple) -> {  // ROUND 2 : Map
                    Long i = tuple._1()._1();
                    Long j = tuple._1()._2();

                    return LongStream
                            .range(0, sqrt_of_N)
                            .map(j_new -> j * sqrt_of_N + j_new)
                            .mapToObj(j_new -> // ROUND 3 : Map
                                    new Tuple2<>(new Tuple2<>(Long.min(i, j_new), Long.max(i, j_new)), tuple._2())
                            )
                            .iterator();
                })
                .groupByKey()
                .mapValues(values -> {
                    Iterator<Long> vals = values.iterator();
                    Long i = vals.next();
                    if (vals.hasNext())
                        return Math.abs(i - vals.next());
                    else
                        return 0L;
                })
                .mapToPair(tuple -> {
                    Long i = tuple._1()._1();
                    Long j = tuple._1()._2();
                    Long new_key = (i * sqrt_of_N + j) % Math.round(Math.pow(sqrt_of_N.doubleValue(), 1.5));
                    return new Tuple2<>(new_key, tuple._2());
                })
                .reduceByKey(Long::max)
                .mapToPair(tuple -> new Tuple2<>(tuple._1() % N, tuple._2()))
                .reduceByKey(Long::max)
                .mapToPair(tuple -> new Tuple2<>(tuple._1() % sqrt_of_N, tuple._2()))
                .reduceByKey(Long::max)
                .values()
                .reduce(Long::max);

        System.out.println("MAXIMUM PAIRWISE DISTANCE in R = O(1), M_L = O(N^1/2), M_A = (N^2):  " + max_length);

        /// MAXIMUM PAIRWISE DISTANCE in R = O(N^1/2), M_L = O(N^1/2), M_A = (N)

        JavaPairRDD<Long, Long> rdd = dataset
                .zipWithIndex()
                .mapToPair((tuple) -> new Tuple2<>(tuple._2() % sqrt_of_N, Long.valueOf(tuple._1())))
                .groupByKey()
                .flatMapToPair((t) -> {
                    List<Tuple2<Long, Long>> values = new ArrayList<>();
                    t._2().forEach(v -> values.add(new Tuple2<>(t._1, v)));

                    long max = 0L;
                    for (int i = 0; i < values.size(); i++)
                        for (int j = i + 1; j < values.size(); j++)
                            max = Math.max(max, Math.abs(values.get(i)._2() - values.get(j)._2()));
                    values.add(new Tuple2<>(-1L, max));
                    return values.iterator();
                });

        for (int iteration = 1; iteration < sqrt_of_N; iteration++) {
            int finalIteration = iteration;
                    rdd = rdd.flatMapToPair((t) -> {
                        if (t._1 >= 0) {
                            return Stream.of(Tuple2.apply(t._1, Tuple2.apply(t._2, true)),
                                    Tuple2.apply((t._1 + finalIteration) % sqrt_of_N, Tuple2.apply(t._2, false)))
                                    .iterator();
                        } else {
                            return Stream.of(Tuple2.apply(t._1, Tuple2.apply(t._2, true)))
                                    .iterator();
                        }
                    })
                    .groupByKey()
                    .flatMapToPair(t -> {
                        if (t._1().equals(-1L)) {
                            Iterator<Tuple2<Long, Boolean>> it = t._2().iterator();
                            long max = 0;
                            while (it.hasNext())
                                max = Math.max(max, it.next()._1());
                            return Stream.of(Tuple2.apply(-1L, max)).iterator();
                        } else {
                            List<Tuple2<Long, Long>> original = new ArrayList<>();
                            List<Long> imported = new ArrayList<>();
                            t._2().forEach(v -> {
                                if (v._2())
                                    original.add(Tuple2.apply(t._1, v._1()));
                                else
                                    imported.add(v._1());
                            });
                            long max = 0;
                            for (Tuple2<Long, Long> o : original)
                                for (long i : imported)
                                    max = Math.max(max, Math.abs(o._2 - i));
                            original.add(Tuple2.apply(-1L, max));
                            return original.iterator();
                        }
                    });
        }

        max_length = rdd.filter(t -> t._1.equals(-1L))
                .values()
                .reduce(Math::max);

        System.out.println("MAXIMUM PAIRWISE DISTANCE in R = O(N^1/2), M_L = O(N^1/2), M_A = (N):  " + max_length);

    }
}
