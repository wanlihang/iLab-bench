import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PageRank {
    public static void main(String[] args) {
        // 创建SparkConf
        SparkConf conf = new SparkConf()
            .setAppName("PageRank")
            .setMaster("local");

        // 创建javaSparkContext
        JavaSparkContext sc = new JavaSparkContext(conf);

        // 读取文件
        JavaRDD<String> data = sc.textFile("./data/web-Google.txt");

        //加载所有的网页关系数据，并且将其转化为pair类型，
        JavaPairRDD<String, List<String>> links = data.mapToPair(x -> {
            String[] a = x.split("\\s+", 2);
            String[] b = a[1].split(",");
            return new Tuple2<String, List<String>>(a[0], Arrays.asList(b));
        });

        //初始化rank 设置每一个页面的初始权重为1.0，使用mapValue生成RDD
        JavaPairRDD<String, Double> ranks = links.mapValues(x -> 1.0);

        //迭代计算更新每个页面的rank，迭代次数可以自由设定，最好是设置结束条件：收敛结束
        for (int i = 0; i < 10; i++) {
            JavaPairRDD<String, Tuple2<List<String>, Double>> tmp = links.join(ranks);
            JavaPairRDD<String, Double> contributions = tmp.flatMapToPair(x -> {
                List<String> ls = x._2._1;
                Double rank = x._2._2;

                List<Tuple2<String, Double>> ret = new ArrayList<>();
                for (String dest : ls) {
                    ret.add(new Tuple2<>(dest, rank / ls.size()));
                }
                return ret.iterator();
            });

            ranks = contributions.reduceByKey(Double::sum).mapValues(v -> 0.15 + 0.85 * v);
        }

        //输出所有的页面的pageRank 值
        List<Tuple2<String, Double>> out = ranks.collect();
        for (Tuple2<String, Double> tuple : out) {
            System.out.println(tuple._1 + "\t " + tuple._2);
        }
        sc.stop();
    }
}
