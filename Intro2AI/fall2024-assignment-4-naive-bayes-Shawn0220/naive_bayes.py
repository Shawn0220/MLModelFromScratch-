import math
import re
from collections import defaultdict

class Bayes_Classifier:
    def __init__(self):
        # 用于存储每个类别的文档数量
        self.class_counts = defaultdict(int)
        # 用于存储每个类别中每个单词的计数
        self.word_counts = defaultdict(lambda: defaultdict(int))
        # 用于存储每个类别的单词总数
        self.total_words_in_class = defaultdict(int)
        # 用于存储训练集中的类别
        self.classes = set()
        # 用于存储训练集中的总单词数量（用于平滑）
        self.vocab = set()

    def train(self, lines):
        """
        训练模型，计算每个类别的先验概率和条件概率。
        :param lines: 训练数据的列表，每个元素是 (类别, 文本) 形式的元组
        """
        formated_lines = []
        for item in lines:
            parts = item.split("|")
            tag = parts[0]
            text = parts[-1]
            formated_lines.append((tag, text))
        # print(formated_lines[0])
        for label, text in formated_lines:
            # 更新类别计数
            self.class_counts[label] += 1
            # 将类别添加到类别集合中
            self.classes.add(label)

            # 拆分文本并计数
            words = text.split()
            for word in words:
                self.word_counts[label][word] += 1
                self.total_words_in_class[label] += 1
                self.vocab.add(word)

    def classify(self, lines):
        """
        对输入文本进行分类。
        :param lines: 需要分类的文本列表
        :return: 返回每个文本的预测类别列表
        """
        results = []
        for text in lines:
            words = text.split()
            class_scores = {}

            # 计算每个类别的概率
            for label in self.classes:
                # 初始化类别的得分为对数形式的先验概率
                log_prob = math.log(self.class_counts[label] / sum(self.class_counts.values()))

                # 计算文本中每个单词的条件概率，并累加到类别得分中
                for word in words:
                    # 使用拉普拉斯平滑
                    word_likelihood = (self.word_counts[label][word] + 1) / (self.total_words_in_class[label] + len(self.vocab))
                    log_prob += math.log(word_likelihood)

                # 存储该类别的得分
                class_scores[label] = log_prob

            # 选择具有最高得分的类别作为预测类别
            best_class = max(class_scores, key=class_scores.get)
            results.append(best_class)
        
        return results