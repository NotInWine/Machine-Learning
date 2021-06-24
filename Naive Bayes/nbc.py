# -*- coding: UTF-8 -*-
# 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
# 优化版多项式朴素贝叶斯
from sklearn.naive_bayes import ComplementNB
# 图形工具
import matplotlib.pyplot as plt
# 系统库
import os
# 随机数库
import random
# 中文分词器
import jieba

"""
函数说明:中文文本处理

Parameters:
	folder_path - 文本存放的路径
	test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
	all_words_list - 按词频降序排序的训练集列表
	train_data_list - 训练集列表
	test_data_list - 测试集列表
	train_class_list - 训练集标签列表
	test_class_list - 测试集标签列表
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-22
"""
def TextProcessing(folder_path, test_size = 0.2):
	base_dir = os.path.dirname(__file__)
	folder_path = os.path.join(base_dir, folder_path)										# 拼接相对路径
	folder_list =  os.listdir(folder_path)													# 拿到文件列表
	data_list = []																			# 数据集数据
	class_list = []																			# 数据集类别

	test_number = {} 																		# 测试集合数量 分类:分类下的测试集合数量							
	for folder in folder_list:																# 遍历每个子文件夹
		new_folder_path = os.path.join(folder_path, folder)									# 根据子文件夹，生成新的路径
		files = os.listdir(new_folder_path)													# 存放子文件夹下的txt文件的列表

		test_number[folder] = int(len(files) * test_size) + 1								# 计算每个分类下的测试集数量
		for file in files:																	# 遍历文件
			with open(
					os.path.join(new_folder_path, file),
				 	'r',
				  	encoding = 'utf-8'
				  ) as f:
				raw = f.read()																# 打开读取os内容
									
			word_cut = jieba.cut(raw, cut_all = False)										# 精简模式，返回一个可迭代的generator
			word_list = list(word_cut)														# generator转换为list
									
			data_list.append(word_list)														# 添加数据集数据
			class_list.append(folder)														# 添加数据集类别
									
	data_class_list = list(zip(data_list, class_list))										# zip压缩合并，将数据与标签对应压缩
	random.shuffle(data_class_list)															# 将data_class_list乱序

	train_list = []																			# 开始填充测试集和训练集
	test_list = []
	for data in data_class_list:
		if test_number[data[1]] > 0:														# 该分类下还有测试集未填充
			test_list.append(data)
			test_number[data[1]] = test_number[data[1]] - 1 
		else :
			train_list.append(data)

	train_data_list, train_class_list = zip(*train_list)									# 训练集解压缩
	test_data_list, test_class_list = zip(*test_list)										# 测试集解压缩
									
	all_words_dict = {}																		# 统计训练集词频
	for word_list in train_data_list:
		for word in word_list:
			if word in all_words_dict.keys():
				all_words_dict[word] += 1
			else:
				all_words_dict[word] = 1

	all_words_tuple_list = sorted(
			all_words_dict.items(),key = lambda f:f[1],reverse = True
		)																					# 根据键的值倒序排序
	all_words_list, all_words_nums = zip(*all_words_tuple_list)								# 解压缩
	all_words_list = list(all_words_list)													# 转换成列表
	return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

"""
函数说明:读取文件里的内容，并去重

Parameters:
	words_file - 文件路径
Returns:
	words_set - 读取的内容的set集合
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-22
"""
def MakeWordsSet(words_file):
	words_set = set()
	base_dir = os.path.dirname(__file__)
	words_file = os.path.join(base_dir, words_file)			#创建set集合
	with open(words_file, 'r', encoding = 'utf-8') as f:		#打开文件
		for line in f.readlines():								#一行一行读取
			word = line.strip()									#去回车
			if len(word) > 0:									#有文本，则添加到words_set中
				words_set.add(word)
	return words_set 											#返回处理结果

"""
函数说明:根据feature_words将文本向量化

Parameters:
	train_data_list - 训练集
	test_data_list - 测试集
	feature_words - 特征集
Returns:
	train_feature_list - 训练集向量化列表
	test_feature_list - 测试集向量化列表
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-22
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
	def text_features(text, feature_words):
		text_words = set(text)
		features = [text.count(word) if word in text_words else 0 for word in feature_words] #出现在特征集中，则置1
		return features
	train_feature_list = [text_features(text, feature_words) for text in train_data_list]
	test_feature_list = [text_features(text, feature_words) for text in test_data_list]
	return train_feature_list, test_feature_list				#返回结果

def TextFeatures1(train_data_list, test_data_list, feature_words):
	def text_features(text, feature_words):
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words] #出现在特征集中，则置1
		return features
	train_feature_list = [text_features(text, feature_words) for text in train_data_list]
	test_feature_list = [text_features(text, feature_words) for text in test_data_list]
	return train_feature_list, test_feature_list				#返回结果


"""
函数说明:文本特征选取

Parameters:
	all_words_list - 训练集所有文本列表
	deleteN - 删除词频最高的deleteN个词
	stopwords_set - 指定的结束语
Returns:
	feature_words - 特征集
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-22
"""
def words_dict(all_words_list, deleteN, stopwords_set = set()):
	feature_words = []							#特征列表
	n = 1
	for t in range(deleteN, len(all_words_list), 1): # 这里有个BUG!!!! 应该先过滤在跳过deleteN个出现频率最高的词
		if n > 2000:							#feature_words的维度为1000
			break
		#如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
		if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5: # 这里有个BUG!!!! 应该先过滤在跳过deleteN个出现频率最高的词
			feature_words.append(all_words_list[t])
		n += 1
	return feature_words

"""
函数说明:新闻分类器

Parameters:
	train_feature_list - 训练集向量化的特征文本
	test_feature_list - 测试集向量化的特征文本
	train_class_list - 训练集分类标签
	test_class_list - 测试集分类标签
Returns:
	test_accuracy - 分类器精度
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-22
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
	classifier = MultinomialNB().fit(train_feature_list, train_class_list)
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	# v = classifier.predict_proba(test_feature_list)
	# print(v)
	return test_accuracy

def TextClassifier1(train_feature_list, test_feature_list, train_class_list, test_class_list):
	classifier = ComplementNB().fit(train_feature_list, train_class_list)
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	# v = classifier.predict_proba(test_feature_list)
	# print(v)
	return test_accuracy

if __name__ == '__main__':
	#文本预处理
	folder_path = 'SogouC\\Sample'				#训练集存放地址

	# 生成stopwords_set
	stopwords_file = os.path.join(os.path.dirname(__file__), 'stopwords_cn.txt')
	stopwords_set = MakeWordsSet(stopwords_file)


	test_accuracy_list = []
	test_accuracy_list1 = []
	test_accuracy_list2 = []
	run = range(0, 150, 1)				#0 20 40 60 ... 980， 要跳过前多少的词，从零开始 每次跳跃20个 直到 1000
	for r in run:
		# 全量词典(按出现频率排序后)，特征集，测试集，特征集分类，测试分类
		all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, 0.01)
		# 跳过前n个
		deleteN = 700
		feature_words = words_dict(all_words_list, deleteN, stopwords_set)  # 过滤后的特征集
		train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words) # 根据过滤后的特征集把 测试集和特征集 坐标化
		test_accuracy_list.append(TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list))
		test_accuracy_list2.append(TextClassifier1(train_feature_list, test_feature_list, train_class_list, test_class_list))

		train_feature_list1, test_feature_list1 = TextFeatures1(train_data_list, test_data_list, feature_words) # 根据过滤后的特征集把 测试集和特征集 坐标化
		test_accuracy_list1.append(TextClassifier(train_feature_list1, test_feature_list1, train_class_list, test_class_list))

	# ave = lambda c: sum(c) / len(c)
	# print(ave(test_accuracy_list))

	plt.plot(run, test_accuracy_list, color='b', label='add', linestyle='--')
	plt.plot(run, test_accuracy_list1, color='r', label='one', linestyle='--')
	plt.plot(run, test_accuracy_list2, color='k', label='super', linestyle='--')
	plt.title('Out Top-n and accuracy')
	plt.xlabel('Out Top-n')
	plt.ylabel('Accuracy')
	plt.legend() # 显示图例
	plt.show()
