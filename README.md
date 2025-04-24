#  Lightweight框架
增加自定义数据测试功能文件夹名字，必须按照如下格式进行测试：
~~~
myselfdateset
├── train
│   ├── ID1
│   └── ID2
├── test
    ├── ID1
    └── ID2
~~~

ID下面放的是图片，具体的数据处理在LightMBN/data_v2/datasets/image/myself.py下面
因为实际需求没有专门的查询集和图库集，于是采用测试集进行自比对的方式进行测试




