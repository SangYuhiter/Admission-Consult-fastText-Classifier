# -*- coding: utf-8 -*-
"""
@File  : DataPretreatment.py
@Author: SangYu
@Date  : 2019/4/27 9:22
@Desc  : 分类数据预处理
"""
import os
import shutil
import jieba


def raw_data_copy():
    """
    复制文件到相应的标签文件夹，只需更改data_statistics里的配置即可
    :return:
    """
    source_all_dir = "../Cluster/cluster/question/source_all"
    raw_classifier_data_dir = "Data"
    raw_classifier_data_statistics = "data_statistics"
    with open(raw_classifier_data_statistics, "r", encoding="utf-8") as f_label_file:
        for line in f_label_file:
            file_dir = line.strip().split("\t")[0]
            label_files = line.strip().split("\t")[-1].split(" ")
            # 创建文件夹
            if not os.path.exists(os.path.join(raw_classifier_data_dir, file_dir)):
                os.mkdir(os.path.join(raw_classifier_data_dir, file_dir))
            for file in label_files:
                shutil.copy(os.path.join(source_all_dir, file), os.path.join(raw_classifier_data_dir, file_dir, file))
            print(file_dir)
            print(label_files)


def label_name_map():
    """
    根据预设文件标签生成标签映射关系
    :return:
    """
    label_name_map_file = "label_name_map"
    raw_classifier_data_statistics = "data_statistics"
    with open(raw_classifier_data_statistics, "r", encoding="utf-8") as f_label_file, \
            open(label_name_map_file, "w", encoding="utf-8") as file_label_map:
        i = 1
        for line in f_label_file:
            file_label_map.write("__label__" + str(i) + "\t" + line.strip().split("\t")[0] + "\n")
            i += 1


def load_label_name_map():
    """
    加载标签名，标签映射关系
    :return:
    """
    label_to_name = {}
    name_to_label = {}
    label_name_map_file = "label_name_map"
    with open(label_name_map_file, "r", encoding="utf-8") as f_label_map:
        for line in f_label_map:
            label = line.strip().split("\t")[0]
            name = line.strip().split("\t")[-1]
            label_to_name[label] = name
            name_to_label[name] = label
    return label_to_name, name_to_label


def load_stop_word_list(file_path):
    """
    加载停用词表
    :param file_path: 停用词表路径
    :return:
    """
    stop_words = set()
    with open(file_path, "r", encoding="utf-8") as f_stopwords:
        for line in f_stopwords:
            stop_words.add(line.strip())
    return stop_words


def data_aggregate():
    """
    数据聚合，将分散的文件聚合成__label__x.train
    :return:
    """
    raw_classifier_data_dir = "Data"
    name_to_label = load_label_name_map()[-1]
    i = 1
    for file_dir in name_to_label.keys():
        file_dir_path = os.path.join(raw_classifier_data_dir, file_dir)
        file_list = os.listdir(file_dir_path)
        print(file_dir_path)
        f_train = open(os.path.join(file_dir_path, "__label__" + str(i) + ".train"), "w", encoding="utf-8")
        for file in file_list:
            # 原始数据
            if "." not in file:
                with open(os.path.join(file_dir_path, file), "r", encoding="utf-8") as f_source:
                    for line in f_source:
                        f_train.write(line)
        f_train.close()
        i += 1


def data_pretreatment():
    """
    数据预处理，读取相应标签下__label__x.train,__label__x.test，分词，去停用词，打标签，生成符合条件的训练文本
    :return:
    """
    # 加载停用词表
    stop_words = load_stop_word_list("stopwords.txt")

    # 读取标签文件夹
    raw_classifier_data_dir = "Data"
    name_to_label = load_label_name_map()[-1]
    i = 1
    for file_dir in name_to_label.keys():
        file_dir_path = os.path.join(raw_classifier_data_dir, file_dir)
        # 训练数据
        with open(os.path.join(file_dir_path, "__label__" + str(i) + ".train"), "r", encoding="utf-8") as f_train, \
                open(os.path.join(file_dir_path, "ft_" + str(i) + ".train"), "w", encoding="utf-8") as f_ft_train:
            for line in f_train:
                seg_line = jieba.cut(line.strip())
                add_str = ""
                for word in seg_line:
                    if word not in stop_words:
                        add_str += word + " "
                f_ft_train.write("__label__" + str(i) + " , " + add_str.strip() + "\n")

        # 测试数据
        with open(os.path.join(file_dir_path, "__label__" + str(i) + ".test"), "r", encoding="utf-8") as f_train, \
                open(os.path.join(file_dir_path, "ft_" + str(i) + ".test"), "w", encoding="utf-8") as f_ft_train:
            for line in f_train:
                seg_line = jieba.cut(line.strip())
                add_str = ""
                for word in seg_line:
                    if word not in stop_words:
                        add_str += word + " "
                f_ft_train.write("__label__" + str(i) + " , " + add_str.strip() + "\n")
        i += 1


def data_all_aggregate():
    """
    聚合每一类的fasttext训练，测试数据
    :return:
    """
    raw_classifier_data_dir = "Data"
    name_to_label = load_label_name_map()[-1]
    ft_train = open("fasttext.train", "w", encoding="utf-8")
    ft_test = open("fasttext.test", "w", encoding="utf-8")
    i = 1
    for file_dir in name_to_label.keys():
        file_dir_path = os.path.join(raw_classifier_data_dir, file_dir)
        file_list = os.listdir(file_dir_path)
        print(file_dir_path)
        with open(os.path.join(file_dir_path, "ft_"+str(i)+".train"),"r",encoding="utf-8") as f_train:
            for line in f_train:
                ft_train.write(line)
        with open(os.path.join(file_dir_path, "ft_"+str(i)+".test"),"r",encoding="utf-8") as f_test:
            for line in f_test:
                ft_test.write(line)
        i += 1
    ft_train.close()
    ft_test.close()


if __name__ == '__main__':
    # raw_data_copy()
    # label_name_map()
    # map1, map2 = load_label_name_map()
    # print(map1)
    # print(map2)
    # load_stop_word_list("stopwords.txt")
    # data_aggregate()
    # data_pretreatment()
    data_all_aggregate()
    # print(list(jieba.cut("年后发法撒旦就是快乐的方式地方撒旦分数\n")))
