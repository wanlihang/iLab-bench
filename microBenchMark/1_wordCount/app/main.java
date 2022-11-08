import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import scala.Tuple2;
/**
 * @author: wlh
 */
public class SparkWordCount {
    public static void main(String[] args) {
        //    复杂模式
        //    创建SparkConf
        SparkConf conf = new SparkConf();
        conf.setAppName("spark_demo_java");
        conf.setMaster("local");

        //    创建javaSparkContext
        JavaSparkContext sc = new JavaSparkContext(conf);
        //    读取文件
        JavaRDD<String> lines = sc.textFile("C:/dataset/shakespeare/t8.shakespeare.txt");
        //    截取单词
        JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterator<String> call(String s) throws Exception {
                return Arrays.asList(s.split("\\s+")).iterator();
            }
        });
        //    对单词进行计数
        JavaPairRDD<String, Integer> pairWord = words.mapToPair(new PairFunction<String, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String s) throws Exception {
                return new Tuple2<>(s, 1);
            }
        });

        //    根据key进行计算
        JavaPairRDD<String, Integer> result = pairWord.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer i, Integer i2) throws Exception {
                return i + i2;
            }
        });

        //打印结果
        result.foreach(new VoidFunction<Tuple2<String, Integer>>() {
            @Override
            public void call(Tuple2<String, Integer> stringIntegerTuple2) throws Exception {
                System.out.println(stringIntegerTuple2);
            }
        });
        sc.stop();
    }
}
