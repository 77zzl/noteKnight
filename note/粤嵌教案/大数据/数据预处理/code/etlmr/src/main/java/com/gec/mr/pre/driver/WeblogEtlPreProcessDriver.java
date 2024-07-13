package com.gec.mr.pre.driver;

import com.gec.mr.pre.mapper.WeblogPreProcessMapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

public class WeblogEtlPreProcessDriver {

    static {
        try {
            // 设置 HADOOP_HOME 目录
            System.setProperty("hadoop.home.dir", "D:/winutils-master/winutils-master/hadoop-3.0.0/");
            // 加载库文件
            System.load("D:/winutils-master/winutils-master/hadoop-3.0.0/bin/hadoop.dll");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" + e);
            System.exit(1);
        }
    }


    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration configuration = new Configuration();
        Job job = Job.getInstance(configuration);

        FileInputFormat.addInputPath(job,new Path("file:///d:\\src\\input"));
        job.setInputFormatClass(TextInputFormat.class);
        FileOutputFormat.setOutputPath(job,new Path("file:///d:\\src\\weblogPreOut2"));
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setJarByClass(WeblogEtlPreProcessDriver.class);

        job.setMapperClass(WeblogPreProcessMapper.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setNumReduceTasks(0);
        boolean res = job.waitForCompletion(true);
    }
}
