import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;


public class Average {

    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]")
                .setAppName("Average");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> dataset = sc.textFile("resources/max_pairwise_distance.txt").repartition(4);
        long N = dataset.count();
        long sqrt_of_N = (long) Math.sqrt((double) N);

        double average = 0.0;

        /// Average in R = O(1), M_L = O(N), M_A = (N)

        average = dataset
                .map(Long::valueOf)
                .reduce(Long::sum)
                .doubleValue() / N;

        System.out.println("Average in R = O(1), M_L = O(N), M_A = (N): " + average);

        /// Average in R = O(1), M_L = O(N^1/2), M_A = (N)

        average = dataset
                .map(Long::valueOf)
                .zipWithIndex()
                .mapToPair(t -> new Tuple2<>(t._2() % sqrt_of_N, t._1))
                .reduceByKey(Long::sum)
                .values()
                .reduce(Long::sum)
                .doubleValue() / N;

        System.out.println("Average in R = O(1), M_L = O(N^1/2), M_A = (N) " + average);
    }
}
