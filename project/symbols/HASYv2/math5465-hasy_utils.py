import csv
import os
import time
import numpy as np
import graphlearning as gl
from sklearn.svm import SVC
from PIL import Image


def _load_csv(filepath, delimiter=',', quotechar="'"):
	"""
	Load a CSV file.

	Parameters
	----------
	filepath : str
		Path to a CSV file
	delimiter : str, optional
	quotechar : str, optional

	Returns
	-------
	list of dicts : Each line of the CSV file is one element of the list.
	path,symbol_id,latex,user_id
	"""
	data = []
	csv_dir = os.path.dirname(filepath)
	with open(filepath, 'r') as csvfile:
		reader = csv.DictReader(csvfile,
								delimiter=delimiter,
								quotechar=quotechar)
		for row in reader:
			if 'path' in row:
				row['path'] = os.path.abspath(os.path.join(csv_dir,
														   row['path']))
			data.append(row)
	return data


def generate_index(data):
	"""
	Generate an index 0...k for the k labels.

	Parameters
	----------
	csv_filepath : str
		Path to 'test.csv' or 'train.csv'

	Returns
	-------
	dict : Maps a symbol_id as in test.csv and
		train.csv to an integer in 0...k, where k is the total
		number of unique labels.
	"""
	symbol_id2index = {}
	i = 0
	for data_item in data:
		symbol_id = data_item["symbol_id"]
		try:
			data_item["newSymbol_id"] = symbol_id2index[symbol_id]
		except KeyError:
			symbol_id2index[symbol_id] = i
			data_item["newSymbol_id"] = i
			i += 1
	return data,symbol_id2index


def update_old_index(data,new_symbol_index):
	"""
	Unknown labels get appended to new_symbol_index.
	"""
	max_index = max(new_symbol_index.values())
	for i,data_item in enumerate(data):
		symbol_id = data_item["symbol_id"]
		try:
			data_item["newSymbol_id"] = new_symbol_index[symbol_id]
		except KeyError:
			max_index +=1
			new_symbol_index[symbol_id] = max_index
			data_item["newSymbol_id"] = max_index
	return data,new_symbol_index


def update_old_index_and_reject(data,new_symbol_index):
	"""
	Data with unknown labels get discarded.
	"""
	for i,data_item in enumerate(data):
		symbol_id = data_item["symbol_id"]
		try:
			data_item["newSymbol_id"] = new_symbol_index[symbol_id]
		except KeyError:
			data.pop(i)
	return data,new_symbol_index


def get_data_and_labels(csv_filepath, data, one_hot=False):
	"""
	Load the images into a numpy array

	Parameters
	----------
	csv_filepath : str
		'test.csv' or 'train.csv'
	data : list of dicts
		keys: "path","symbol_id","latex","user_id","newSymbol_id"
	one_hot : bool, optional
		Make label vector as 1-hot encoding, otherwise index

	Returns
	-------
	images, labels
	"""
	WIDTH, HEIGHT = 32, 32
	dataset_path = os.path.dirname(csv_filepath)  # Parent directory of csv_filepath
	images = np.zeros((len(data), WIDTH * HEIGHT))
	labels = []
	for i, data_item in enumerate(data):
		fname = os.path.join(dataset_path, data_item['path'])
		img = np.asarray(Image.open(fname).convert('1'))
		images[i, :] = img.flatten()
		label = data_item["newSymbol_id"]
		labels.append(label)
	data = images, np.array(labels)
	if one_hot:
		data = (data[0], np.eye(data[1].max()+1)[data[1]])
	return data


def numpy_load(file, field):
	"""Load an array from a numpy file
	======

	Loads a numpy .npz file and returns a specific field.

	Parameters
	----------
	file : string
		Namename of .npz file
	field : string
		Name of field to load
	"""

	try:
		M = np.load(file,allow_pickle=True)
		d = M[field]
	except:
		sys.exit('Error: Cannot open '+file+'.')

	return d


def get_np_data_and_labels(csv_filepath, data, one_hot=False):
	"""
	Load the images into a numpy array

	Parameters
	----------
	csv_filepath : str
		'test.csv' or 'train.csv'
	data : list of dicts
		keys: "path","symbol_id","latex","user_id","newSymbol_id"
	one_hot : bool, optional
		Make label vector as 1-hot encoding, otherwise index

	Returns
	-------
	images, labels
	"""
	WIDTH, HEIGHT = 32, 32
	dataset_path = os.path.dirname(csv_filepath)  # Parent directory of csv_filepath
	prefix,suffix = os.path.split(csv_filepath)
	suffix = suffix.split('.')[0]+".npz"
	numpy_data = numpy_load(os.path.join(prefix,suffix),"data")
	labels = []
	for data_item in data:
		label = data_item["newSymbol_id"]
		labels.append(label)
	data = numpy_data, np.array(labels)
	if one_hot:
		data = (data[0], np.eye(data[1].max()+1)[data[1]])
	return data


def numpify_and_save_data_helper(csv_filepath, delimiter=',', quotechar="'"):
	"""
	Transform image into a numpy array and then save it.

	Parameters
	----------
	csv_filepath : str
		csv file of data
	"""
	WIDTH, HEIGHT = 32, 32
	csv_dir = os.path.dirname(csv_filepath)  # Parent directory of csv_filepath
	num_lines = 0
	with open(csv_filepath, 'r') as csvfile:
		num_lines = len(csvfile.readlines())-1
	images = np.zeros((num_lines, WIDTH * HEIGHT))
	with open(csv_filepath, 'r') as csvfile:
		reader = csv.DictReader(csvfile,delimiter=delimiter,quotechar=quotechar)
		for i,row in enumerate(reader):
			if 'path' in row:
				fname_old = os.path.abspath(os.path.join(csv_dir, row['path']))
				img = np.asarray(Image.open(fname_old).convert('1'))
				images[i, :] = img.flatten()
	prefix,suffix = os.path.split(csv_filepath)
	suffix = suffix.split('.')[0]+".npz"
	np.savez_compressed(os.path.join(prefix,suffix),data=images)
	return 0


def numpify_and_save_data():
	FILEPATH_TRAIN = "classification-task/fold-1/train.csv"
	FILEPATH_TEST = "classification-task/fold-1/test.csv"
	numpify_and_save_data_helper(FILEPATH_TRAIN)
	numpify_and_save_data_helper(FILEPATH_TEST)
	return 0


def test_svm(get_testing_accuracy=False):
	time_start = time.time()
	FILEPATH_TRAIN = "classification-task/fold-1/train.csv"
	FILEPATH_TEST = "classification-task/fold-1/test.csv"

	# load csv
	data_train = _load_csv(FILEPATH_TRAIN)
	data_test = _load_csv(FILEPATH_TEST)
	print("COMPLETED: CSV FILES LOADED")

	# generate new symbol index and IDs
	data_train,new_symbol_index = generate_index(data_train)
	data_test,new_symbol_index = update_old_index(data_test,new_symbol_index)
	print("COMPLETED: NEW SYMBOL IDs")

	# load data and labels
	x_train,y_train = get_np_data_and_labels(FILEPATH_TRAIN,data_train)
	print("COMPLETED: TRAINING DATA LOADED")
	x_test,y_test = get_np_data_and_labels(FILEPATH_TEST,data_test)
	print("COMPLETED: TESTING DATA LOADED")


	#Train SVM
	#clf = SVC(kernel='linear')
	#clf = SVC()	#75.05%
	clf = SVC(kernel='poly')	#73.08%
	clf.fit(x_train,y_train)
	print("COMPLETED: SVM TRAINING")

	#Training accuracy
	if (get_testing_accuracy):
		y_pred = clf.predict(x_train)
		train_acc = np.mean(y_pred == y_train)*100
		print('SVM Training Accuracy: %.2f%%'%train_acc)

	#Testing accuracy
	y_pred = clf.predict(x_test)
	test_acc = np.mean(y_pred == y_test)*100
	print('SVM Testing Accuracy: %.2f%%'%test_acc)

	num_show = 20
	width, height = 32, 32
	class_size = len(new_symbol_index)
	img = np.zeros((class_size*num_show,width*height))
	I = y_pred != y_test
	x_wrong = x_test[I]
	y_wrong = y_test[I]
	for i in range(class_size):
		I = y_wrong == i
		img[num_show*i:num_show*i + min(num_show,np.sum(I)),:] = x_wrong[I,:][:num_show]

	gl.utils.image_grid(img,n_rows=10,n_cols=num_show)
	print("Execution took %s minutes." % ((time.time()-time_start)/60))
	return 0
