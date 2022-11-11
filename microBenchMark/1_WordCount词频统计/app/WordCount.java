import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import scala.Tuple2;

import java.util.Arrays;

public class WordCount {
    public static void main(String[] args) {
        // 复杂模式
        // 创建SparkConf
        SparkConf conf = new SparkConf()
            .setAppName("WordCount")
            .setMaster("local");

        // 创建javaSparkContext
        JavaSparkContext sc = new JavaSparkContext(conf);

        // 读取文件
        JavaRDD<String> lines = sc.textFile("microBenchMark/1_WordCount词频统计/app/data/t8.shakespeare.txt");

        // 截取单词
        JavaRDD<String> words = lines.flatMap((FlatMapFunction<String, String>) s -> Arrays.asList(s.split("\\s+")).iterator());

        // 对单词进行计数
        JavaPairRDD<String, Integer> pairWord = words.mapToPair((PairFunction<String, String, Integer>) s -> new Tuple2<>(s, 1));

        // 根据key进行计算
        JavaPairRDD<String, Integer> result = pairWord.reduceByKey((Function2<Integer, Integer, Integer>) Integer::sum);

        // 打印结果
        result.foreach((VoidFunction<Tuple2<String, Integer>>) System.out::println);

        sc.stop();
    }
}
