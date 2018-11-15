import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection
from sklearn import tree


def input_data():

	path = r'./fake-real-news/train/fake/'
	txt_files = glob.glob(path + "*.txt")
	
	# with open(txt_files[0], 'r') as myfile:
	# 	data = myfile.read().replace('\n', '')

	txt_df = pd.DataFrame(columns = ['word', 'truth'])	

	limit = 100

	for file in txt_files:
		with open(file, 'r') as f:
			limit -= 1
			if limit == 0:
				break
			text = f.read().replace('\n', '')								
			txt_df = txt_df.append(pd.DataFrame({'word': [text], 'truth': 0}))

	# truth: {0 == Fake, 1 == Real}

	path = r'./fake-real-news/train/real/'
	txt_files = glob.glob(path + "*.txt")


	limit = 100
	for file in txt_files:
		with open(file, 'r') as f:
			limit -= 1
			if limit == 0:
				break
			text = f.read().replace('\n', '')								
			txt_df = txt_df.append(pd.DataFrame({'word': [text], 'truth': 1}))

	txt_df = txt_df.reset_index(drop=True)	

	# txt_df.word = txt_df.word.apply(word_tokenize)
	
	return txt_df
	

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

	txt_df = input_data()
	training_model(txt_df)