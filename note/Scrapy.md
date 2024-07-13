# Scrapy

### 开启一个项目

```
scrapy startproject tutorial
```

<br>

### 目录

```
tutorial/
    scrapy.cfg            # deploy configuration file

    tutorial/             # project's Python module, you'll import your code from here
        __init__.py

        items.py          # project items definition file

        middlewares.py    # project middlewares file

        pipelines.py      # project pipelines file

        settings.py       # project settings file

        spiders/          # a directory where you'll later put your spiders
            __init__.py
```

<br>

### 更改配置文件

```python
# 开启以下注释并修改

DEFAULT_REQUEST_HEADERS = {
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'en',
    'User-Agent': 'Mozilla/5.0 (Windows NT; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36 TXEduLite/40142 (1.1.2.37) Theme/1.0 (NewUI)'
}

ITEM_PIPELINES = {
   'mySpider.pipelines.MyspiderPipeline': 300,
}
```

<br>

### Item

```python
# 创建你需要的类成员
import scrapy


class MyspiderItem(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.Field()
    level = scrapy.Field()
```

<br>

### Generate

#### Spiders

现在终端输入如下命令自动生成爬虫文件`scrapy genspider --name --url`

- name：爬虫名，必须唯一
- url：必须为字符串，表示要爬取的域名地址

#### Crawl

```python
# -*- coding: utf-8 -*-
import scrapy
from mySpider.items import MyspiderItem


class LcSpider(scrapy.Spider):
    # 爬虫名唯一
    name = 'lc'
    # 要爬取的域名
    allowed_domains = ['leetcode.cn']
    # 开始爬取的页面
    start_urls = ['https://leetcode.cn/problemset/all/']

    def parse(self, response):
        datas =response.xpath('//div[@role="rowgroup"]/div[@role="row"]')
        item = MyspiderItem()

        for data in datas:
            title = data.xpath('./div[@role="cell"][2]//a/text()').extract()
            title = title[0] + '.' + title[-1]
            level = data.xpath('./div[@role="cell"][last()-1]/span/text()').extract()[0]
            item['title'], item['level'] = title, level

            yield item

```

<br>

### Pipeline

```python
# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import json


class MyspiderPipeline:
    # 开始爬虫
    def open_spider(self, spider):
        self.filename = open('leetcode.json', 'w', encoding='utf-8')

    # 爬虫进行中
    def process_item(self, item, spider):
        json_str = json.dumps(dict(item), ensure_ascii=False)
        # json_str = json_str.encode().decode()
        self.filename.write(json_str + '\n')
        return item

    # 结束爬虫
    def close_spider(self, spider):
        self.filename.close()

```

### Run

```
scrapy crawl lc
```

