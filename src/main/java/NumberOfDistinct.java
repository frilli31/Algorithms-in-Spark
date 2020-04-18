import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.HashSet;


public class NumberOfDistinct {

    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]")
                .setAppName("NumberOfDistinct");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> dataset = sc.textFile("resources/max_pairwise_distance.txt").repartition(4);
        long N = dataset.count();
        long sqrt_of_N = (long) Math.sqrt((double) N);

        long distinct;

        /// Number of Distinct in R = O(1), M_L = O(N), M_A = (N)

        distinct = dataset
                .map(Long::valueOf)
                .distinct()
                .count();

        System.out.println("Number of Distinct in R = O(1), M_L = O(N), M_A = (N): " + distinct);


        /// Average in R = O(1), M_L = O(N^1/2), M_A = (N)

        distinct = dataset
                .map(Long::valueOf)
                .zipWithIndex()
                .mapToPair(t -> new Tuple2<>(t._2 % sqrt_of_N, t._1))
                .groupByKey()
                .flatMapValues(t -> {
                    HashSet<Long> accumulator = new HashSet<Long>();
                    t.forEach(accumulator::add);
                    return accumulator;
                })
                .mapToPair(t -> Tuple2.apply(t._2, t._1))
                .reduceByKey((x, y) -> x)
                .mapToPair(t -> Tuple2.apply(t._2, t._1))
                .aggregateByKey(0L, (acc, el) -> acc + 1, Long::sum)
                .values()
                .reduce(Long::sum);

        System.out.println("Number of Distinct in R = O(1), M_L = O(N^1/2), M_A = (N): " + distinct);


        /// Average in R = O(1), M_L = O(N^1/2), M_A = (N)


    }
}
