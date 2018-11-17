import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection
from sklearn import tree
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import movie_reviews
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def input_data():

	path = r'./fake-real-news/train/fake/'
	txt_files = glob.glob(path + "*.txt")
	
	
	limit = 10000

	stop_words = set(stopwords.words('english'))

	tokenizer = RegexpTokenizer(r'\w+')

	X = []
	Y = []

	word_dis = FreqDist()

	for file in txt_files:
		with open(file, 'r', encoding="utf8") as f:
			limit -= 1
			if limit == 0:
				break
			text = f.read().replace('\n', '')				
			text = tokenizer.tokenize(text)

			words = []

			for w in text:
				if w not in stop_words:
					words.append(w)			

			for word in words:
				word_dis[word.lower()] += 1

			X.append(words)
			Y.append([0])
			

	#==========================================#

	# truth: {0 == Fake, 1 == Real}

	path = r'./fake-real-news/train/real/'
	txt_files = glob.glob(path + "*.txt")

	limit = 10000

	for file in txt_files:
		with open(file, 'r', encoding="utf8") as f:
			limit -= 1
			if limit == 0:
				break
			text = f.read().replace('\n', '')				
			text = tokenizer.tokenize(text)

			words = []

			for w in text:
				if w not in stop_words:
					words.append(w)			

			for word in words:
				word_dis[word.lower()] += 1

			X.append(words)
			Y.append([1])

	#========================================

	feature_limit = 500
	word_features = []
	for word_feature in word_dis.most_common(feature_limit):
		word_features.append(word_feature[0])
		
	features_list = []
	x = []
	for i in range(len(X)):
		features = {}
		temp_x = []
		for word in word_features:
			features['contains({})'.format(word)] = (word in X[i])
			temp_x.append(int(word in X[i]))
		x.append(temp_x)
		features_list.append(features)

	x = np.array(x)
	y = np.array(Y).ravel()


	accuracy_list = []
	x_list = []


	################## Option 1 Different node in layer 1 ##################

	# layer_1 = 2
	# while layer_1 <= 100:

	# 	clf = MLPClassifier(solver='lbfgs', \
	# 							alpha=1e-5, \
	# 							hidden_layer_sizes=(layer_1, 2), \
	# 							random_state=1, \
	# 							learning_rate_init=0.25)
	# 	acc = training(clf, 10, x, y)

	# 	accuracy_list.append(acc)
	# 	x_list.append(layer_1)
	# 	print("layer_1 = ", layer_1, "accuracy = ", acc)
	# 	layer_1 += 1

	# result = pd.DataFrame({'layer_1': x_list, 'accuracy': accuracy_list})

	# ax = sns.lineplot(x="layer_1", y="accuracy", data=result)
	# plt.show()

	########################################################################


	################### Option 2 Different learning rate ###################

	learning_rate = 0.01
	while learning_rate <= 1:

		clf = MLPClassifier(solver='lbfgs', \
								alpha=1e-5, \
								hidden_layer_sizes=(3, 20), \
								random_state=1, \
								learning_rate_init=learning_rate)
		acc = training(clf, 10, x, y)

		accuracy_list.append(acc)
		x_list.append(learning_rate)
		print("learning_rate = ", learning_rate, "accuracy = ", acc)
		learning_rate += 0.03

	result = pd.DataFrame({'learning_rate': x_list, 'accuracy': accuracy_list})

	ax = sns.lineplot(x="learning_rate", y="accuracy", data=result)
	plt.show()

	########################################################################

def training(model, n_split, data_x, data_y):
	clf = model
	kf = KFold(n_splits = n_split, shuffle = True)
	x = data_x
	y = data_y

	accuracy = 0

	for train, test in kf.split(x):
		x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
		clf = clf.fit(x_train, y_train)
		accuracy += clf.score(x_test, y_test)

	return accuracy / n_split



if __name__  ==  "__main__":

	input_data()
	# training_model(txt_df)