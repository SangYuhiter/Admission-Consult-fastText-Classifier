# Admission-Consult-fastText-Classifier
招生咨询领域的fastText文本分类

## Data:20类文本，每类文本中包含训练集和测试集（训练集不准确，有噪声；测试集人工挑选，是准确的）
## DataPretreatment.py数据预处理（将文本处理成fastText需要的格式）
## FastTextModel.py训练fastText模型，测试fastText模型

### data_statistics:数据统计结果
### label_name_map:标签名字映射
### stopwords.txt:停用词
### fasttext.train,fasttext.test训练和测试数据
### fasttext-0.8.22-....whl fastText的windows包

#### 使用方法：需要fastText的pypi包（windows）,pip install 即可
#### 修改模型训练中双层循环的参数即可选择相应的ngram值和训练轮数epoch(Classifier下创建Model文件夹保存训练的模型)
#### 模型命名model_wx_ey:ngram=x,epoch=y时训练的模型。
