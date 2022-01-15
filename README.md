# 银行产品评论观点提取
**南京大学计算机科学与技术系研究生选修课程《自然语言处理》（2021秋）课程项目作业**

本项目采用PaddlePaddle+ERNIE进行实现。

本项目可在[百度AI Studio平台](https://aistudio.baidu.com/aistudio/projectdetail/3281489)在线运行和调试（需要使用GPU环境）。项目源码已上传[Github](https://github.com/MilesPoupart/nju-nlp-project-2021)。

项目数据集已包含在本项目`/data/data122751`路径中，具体数据集说明和下载见[数据集详情页面](https://aistudio.baidu.com/aistudio/datasetdetail/122751)。

本项目的代码主要参考了[这一博文](https://blog.csdn.net/weixin_41611054/article/details/118487666)（[GitHub项目地址](https://github.com/zhenhao-huang/paddlehub_ernie_emotion_analysis)），特表示感谢！

## 文件说明
main.ipynb包含所有任务的源码和详细说明。to_csv.py是数据集划分预处理程序源码，finetune_ernie-classify.py和finetune_ernie-ner.py是分类任务和NER任务的加载数据和训练保存模型的代码，predict-classify.py和predict-ner.py是预测程序的源码。

## 项目要求
现有有关银行及银行相关产品的若干条中文评论，要求对这些评论进行观点提取。具体而言，分为以下两个子任务：

### 实体识别
要求识别出原始评论文本中的实体及类型，并按BIO格式进行标注。<br>需要进行识别的实体有：银行、产品、用户评论中的名词及形容词，具体的标注标签及说明如下表所示：

|标签|说明|
|:----:|:----:|
|B-BANK|代表银行实体的开始|
|I-BANK|代表银行实体的内部|
|B-PRODUCT|代表产品实体的开始|
|I-PRODUCT|代表产品实体的内部|
|B-COMMENTS_N|代表用户评论（名词）|
|I-COMMENTS_N|代表用户评论（名词）实体的内部|
|B-COMMENTS_ADJ|代表用户评论（形容词）|
|I-COMMENTS_ADJ|代表用户评论（形容词）实体的内部|
|O|代表不属于标注的范围|

下图是一个标注的例子：
![实体识别的标注举例](https://ai-studio-static-online.cdn.bcebos.com/4966caab43894ab8ad1a7a85bdaff87a8437b4bf20b74ab0ad379c7cdcf1a248)

### 情感分类
根据用户评论的文本内容，判断其情感极性，并对其情感进行分类。<br>本次实验任务中，需要将用户的评论划分为正面（1）、负面（0）和中立（2）三种类型。