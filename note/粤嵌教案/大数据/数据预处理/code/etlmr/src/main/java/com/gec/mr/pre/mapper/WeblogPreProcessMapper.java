package com.gec.mr.pre.mapper;

import com.gec.mr.pre.mrbean.WebLogBean;
import com.gec.mr.pre.utils.WebLogParser;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class WeblogPreProcessMapper extends Mapper<LongWritable, Text, Text, NullWritable>
{
    // 用来存储网站url分类数据
    Set<String> pages = new HashSet<String>();
    Text k = new Text();
    NullWritable v = NullWritable.get();
    /**
     * map阶段的初始化方法
     * 从外部配置文件中加载网站的有用url分类数据 存储到maptask的内存中，用来对日志数据进行过滤
     * 过滤掉我们日志文件当中的一些静态资源，包括js   css  img  等请求日志都需要过滤掉
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        //定义一个集合
        pages.add("/about");
        pages.add("/black-ip-list/");
        pages.add("/cassandra-clustor/");
        pages.add("/finance-rhive-repurchase/");
        pages.add("/hadoop-family-roadmap/");
        pages.add("/hadoop-hive-intro/");
        pages.add("/hadoop-zookeeper-intro/");
        pages.add("/hadoop-mahout-roadmap/");

    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        //得到我们一行数据
        String line = value.toString();
        WebLogBean webLogBean = WebLogParser.parser(line);
        if (webLogBean != null) {
            // 过滤js/图片/css等静态资源
            WebLogParser.filtStaticResource(webLogBean, pages);
            if (!webLogBean.isValid()) return;
            k.set(webLogBean.toString());
            context.write(k, v);
        }
    }
}
