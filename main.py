import pandas as pd
import numpy as np
import glob


def input_data():
	path = r'./fake-real-news/train/fake/'
	txt_files = glob.glob(path + "*.txt")
	print(txt_files)

	for file in txt_files:
		data = pd.read_csv(file, sep = " ", header = None)
		data.columns = ["text", "truth"]
		print(data)
	# l = [pd.read_csv(file_name) for file_name in glob.glob(path + "/*.txt")]
	# df = pd.concat(l, axis = 0)
	# print(df)


if __name__  ==  "__main__":

	input_data()