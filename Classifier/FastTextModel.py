# -*- coding: utf-8 -*-
"""
@File  : FastTextModel.py
@Author: SangYu
@Date  : 2019/4/28 10:57
@Desc  : FastText模型训练
"""
import time
import fastText.FastText as ff
import jieba
from Classifier.DataPretreatment import load_label_name_map,load_stop_word_list
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']


def fasttext_model_train():
    """
    fasttext模型训练
    :return:
    """
    for i in range(5, 51):
        for w in range(1, 3):
            start_time = time.time()
            classifier = ff.train_supervised("fasttext.train", epoch=i, lr=0.5, wordNgrams=w)
            print("ngram=%d,训练第%d轮，用时%s" % (w, i, time.time() - start_time))
            classifier.save_model("Model/model_w" + str(w) + "_e" + str(i))


def load_model_to_test():
    """
    加载模型进行测试
    :return:
    """
    # 加载测试数据
    correct_labels = []
    texts = []
    with open("fasttext.test", "r", encoding="utf-8") as ft_test:
        for line in ft_test:
            correct_labels.append(line.strip().split(" , ")[0])
            texts.append(line.strip().split(" , ")[1])

    # 加载分类模型
    for w in range(1, 2):
        all_marco_precision = []
        all_marco_recall = []
        all_marco_f1 = []
        all_micro_precision = []
        all_micro_recall = []
        all_micro_f1 = []
        for i in range(5, 51):
            classifier = ff.load_model("Model/model_w" + str(w) + "_e" + str(i))
            print("Model/model_w" + str(w) + "_e" + str(i))
            # 预测
            predict_labels = classifier.predict(texts)[0]
            # 计算预测结果
            true_positive = 0
            false_positive = 0
            false_negative = 0
            evaluation_parameters = []
            label_to_name = load_label_name_map()[0]
            for label, name in label_to_name.items():
                evaluate_p = {}
                evaluate_p["name"] = name
                evaluate_p["nexample"] = len(texts)
                for i in range(len(texts)):
                    # 预测属于该类，实际属于该类
                    if predict_labels[i] == label and correct_labels[i] == label:
                        true_positive += 1
                    # 预测属于该类，实际不属于该类
                    elif predict_labels[i] == label and correct_labels[i] != label:
                        false_positive += 1
                    # 预测不属于该类，实际属于该类
                    elif predict_labels[i] != label and correct_labels[i] == label:
                        false_negative += 1
                evaluate_p["true_positive"] = true_positive
                evaluate_p["false_positive"] = false_positive
                evaluate_p["false_negative"] = false_negative
                # 计算精确率、召回率、F值
                precision = true_positive / (true_positive + false_positive)
                evaluate_p["precision"] = precision
                recall = true_positive / (true_positive + false_negative)
                evaluate_p["recall"] = recall
                f1 = 2 * precision * recall / (precision + recall)
                evaluate_p["f1"] = f1
                evaluation_parameters.append(evaluate_p)
                # print("%s标签测试结果：" % name)
                # print("测试集大小：%d\t精确率：%f\t召回率：%f\tF_1：%f" % (len(texts), precision, recall, f1))
            # 计算宏平均和微平均
            sum_precision = 0
            sum_recall = 0
            sum_true_positive = 0
            sum_false_positive = 0
            sum_false_negative = 0
            for p in evaluation_parameters:
                sum_precision += p["precision"]
                sum_recall += p["recall"]
                sum_true_positive += p["true_positive"]
                sum_false_positive += p["false_positive"]
                sum_false_negative += p["false_negative"]
            n = len(evaluation_parameters)
            marco_precision = sum_precision / n
            all_marco_precision.append(marco_precision)
            marco_recall = sum_recall / n
            all_marco_recall.append(marco_recall)
            marco_f1 = 2 * marco_precision * marco_recall / (marco_precision + marco_recall)
            all_marco_f1.append(marco_f1)
            print("宏平均----测试集大小：%d\t精确率：%f\t召回率：%f\tF_1：%f" % (len(texts), marco_precision, marco_recall, marco_f1))
            micro_true_positive = sum_true_positive / n
            micro_false_positive = sum_false_positive / n
            micro_false_negative = sum_false_negative / n
            micro_precision = micro_true_positive / (micro_true_positive + micro_false_positive)
            all_micro_precision.append(micro_precision)
            micro_recall = micro_true_positive / (micro_true_positive + micro_false_negative)
            all_micro_recall.append(micro_recall)
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            all_micro_f1.append(micro_f1)
            print("微平均----测试集大小：%d\t精确率：%f\t召回率：%f\tF_1：%f" % (len(texts), micro_precision, micro_recall, micro_f1))

        names = [i for i in range(5, 51)]
        ax1 = plt.subplot(311)
        plt.plot(names, all_marco_precision, label='marco-P')
        plt.plot(names, all_micro_precision, label='micro-P')
        plt.legend(loc='upper left')
        ax2 = plt.subplot(312, sharey=ax1)
        plt.plot(names, all_marco_recall, label='marco-P')
        plt.plot(names, all_micro_recall, label='micro-R')
        plt.legend(loc='upper left')
        plt.subplot(313, sharey=ax1)
        plt.plot(names, all_marco_f1, label='marco-F1')
        plt.plot(names, all_micro_f1, label='micro-F1')
        plt.legend(loc='upper left')
        plt.xlabel(u"训练轮数(ngram=" + str(w) + ")")
        plt.savefig("./ngram" + str(w) + ".png")
        plt.show()


def question_classifier_test():
    """
    问题分类测试
    :return:
    """
    # 加载停用词表
    stop_words = load_stop_word_list("stopwords.txt")
    label_to_name = load_label_name_map()[0]
    classifier = ff.load_model("Model/model_w2_e24")
    while True:
        input_ = input("question:")
        seg_line = jieba.cut(input_)
        add_str = ""
        for word in seg_line:
            if word not in stop_words:
                add_str += word + " "
        predict = classifier.predict(add_str.strip(), 3)
        print(predict)
        for label in predict[0]:
            print(label_to_name[label])


if __name__ == '__main__':
    # load_model_to_test()
    # fasttext_model_train()
    question_classifier_test()
