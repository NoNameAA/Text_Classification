import pandas as pd
import numpy as np
import glob


def input_data():

	path = r'./fake-real-news/train/fake/'
	txt_files = glob.glob(path + "*.txt")
	
	txt_df = pd.DataFrame(columns = ['word', 'truth'])	

	for file in txt_files:
		with open(file, 'r') as f:
			temp_list = []
			for line in f:
				for word in line.split():
					temp_list.append(word)
		txt_df = txt_df.append(pd.DataFrame({'word': [temp_list], 'truth': 'Fake'}))



	path = r'./fake-real-news/train/real/'
	txt_files = glob.glob(path + "*.txt")

	for file in txt_files:
		with open(file, 'r') as f:
			temp_list = []
			for line in f:
				for word in line.split():
					temp_list.append(word)
		txt_df = txt_df.append(pd.DataFrame({'word': [temp_list], 'truth': 'Real'}))

	txt_df = txt_df.reset_index(drop=True)
	
	print(txt_df)			
	


if __name__  ==  "__main__":

	input_data()