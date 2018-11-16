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

# nltk.download('movie_reviews')

def input_data():

	path = r'./fake-real-news/train/fake/'
	txt_files = glob.glob(path + "*.txt")
	
	# with open(txt_files[0], 'r') as myfile:
	# 	data = myfile.read().replace('\n', '')
	
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
			
	# print(word_dis.most_common(30))	

	# print(Y)

	# print("_______________________")

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

	# print(word_dis.most_common(30))
	feature_limit = 1000
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

	# print(np.shape(x), np.shape(Y))	

	#=======================================

	x = np.array(x)
	y = np.array(Y).ravel()

	# for max_feature in range(10, feature_limit):
	# 	clf = tree.DecisionTreeClassifier(max_features = max_feature)
	# 	acc = training(clf, 15, x, y)
	# 	print("max_feature = ", max_feature, "acc = ", acc)

	#=======================================	

	
	for i in range(2, 30):
		for j in range(2, 30):	
			clf = MLPClassifier(solver='lbfgs', \
								alpha=1e-5, \
								hidden_layer_sizes=(i, j), \
								random_state=1)
			acc = training(clf, 10, x, y)
			print("l1 = ", i, "l2 = ", j, "acc = ", acc)





def training(model, n_split, data_x, data_y):
	clf = model
	kf = KFold(n_splits = n_split, shuffle = True)
	x = data_x
	y = data_y

	accuracy = 0

	for train, test in kf.split(x):
		x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
		clf = clf.fit(x_train, y_train)
		accuracy = max(accuracy, clf.score(x_test, y_test))

	return accuracy

def training_model(txt_df):



	x_train, x_test, y_train, y_test = model_selection.train_test_split( \
										txt_df.word, \
										txt_df.truth, \
										test_size = 0.1)


	# print(x_train)

	count_vect = CountVectorizer()
	x_train_counts = count_vect.fit_transform(x_train)

	# print(x_train_counts)
	tfidf_transformer = TfidfTransformer(use_idf=True)
	x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	print(y_train)
	print(x_train_tfidf)

	clf = tree.DecisionTreeClassifier()			   
	clf = clf.fit(x_train_tfidf, y_train)



if __name__  ==  "__main__":

	input_data()
	# training_model(txt_df)