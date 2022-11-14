package org.example;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        if (0 == args.length) {
            System.out.println("未指定输入文件地址");
            return;
        }

        // 创建SparkConf
        SparkConf conf = new SparkConf()
            .setAppName("WordCount")
            .setMaster("local");

        // 创建javaSparkContext
        JavaSparkContext sc = new JavaSparkContext(conf);

        // 读取文件
        JavaRDD<String> lines = sc.textFile(args[0]);

        // 截取单词
        JavaRDD<String> words = lines.flatMap((FlatMapFunction<String, String>) s -> Arrays.asList(s.split("\\s+")).iterator());

        // 对单词进行计数
        JavaPairRDD<String, Integer> pairWord = words.mapToPair((PairFunction<String, String, Integer>) s -> new Tuple2<>(s, 1));

        // 根据key进行计算
        JavaPairRDD<String, Integer> result = pairWord.reduceByKey((Function2<Integer, Integer, Integer>) Integer::sum);

        // 打印结果
        result.foreach(item -> System.out.println(item._1 + "\t" + item._2));

        sc.stop();
    }
}
