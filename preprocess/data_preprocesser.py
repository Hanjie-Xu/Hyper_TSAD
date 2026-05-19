import os
import sys
import pandas as pd
import numpy as np
import pickle
import json

output_folder = 'processed'
data_folder = 'data'
datasets = ['SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def remove_constants(a, cols=None):
	if cols is None: cols = np.where(a.ptp(0) > 0)[0]
	return a[:, cols], cols

def normalize_z(a, mean_a = None, std_a = None):
	if mean_a is None: mean_a, std_a = np.mean(a, axis = 0), np.std(a, axis = 0)
	return (a - mean_a) / (std_a + 0.0001), mean_a, std_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'UCR':
		dataset_folder = 'data/UCR'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset == 'NAB':
		dataset_folder = 'data/NAB'
		file_list = os.listdir(dataset_folder)
		with open(dataset_folder + '/labels.json') as f:
			labeldict = json.load(f)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			labels = np.zeros_like(vals, dtype=np.float64)
			for timestamp in labeldict['realKnownCause/'+filename]:
				tstamp = timestamp.replace('.000000', '')
				index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
				labels[index-4:index+4] = 1
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train, test = vals.astype(float), vals.astype(float)
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'MSDS':
		dataset_folder = 'data/MSDS'
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
		_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
		train, _, _ = normalize3(df_train, min_a, max_a)
		test, _, _ = normalize3(df_test, min_a, max_a)
		labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
		labels = labels.values[::1, 1:]
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	elif dataset == 'SWaT':
		dataset_folder = 'data/SWaT'
		train = pd.read_csv(os.path.join(dataset_folder, 'swat_train.csv'), nrows=4e5)
		test = pd.read_csv(os.path.join(dataset_folder, 'swat_test.csv'),sep=';', nrows=4e4)
		#train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		#train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		
		def process_swat(ds_orig: pd.DataFrame):
			ds = ds_orig[ds_orig.columns[1:]].copy()
			for col in ds.columns[:-1]:
				if ds[col].dtype == object:
					ds[col] = pd.to_numeric(ds[col].replace(",", ".", regex=True))
			label_col = ds.columns[-1]
			ds[label_col] = (ds[label_col] != "Normal").astype(int) ## 1 for attack, 0 for normal
			return ds[ds.columns[:-1]].values.astype(float), ds[ds.columns[-1]].values.astype(float) 
		train, _ = process_swat(train)
		test, labels_orig = process_swat(test)
		print("orig train shape:", train.shape)
		print("orig test shape:", test.shape)
		print("orig anomaly ratio:", labels_orig.mean())
		labels = test.copy()
		labels[:, :] = 0
		for i in range(test.shape[0]):
			if labels_orig[i] == 1:
				labels[i, :] = 1 ## mark the whole row as anomalous if it is an attack
		train, test, labels = train[::10, :], test[::10, :], labels[::10, :]
		train, cols = remove_constants(train)
		test, _ = remove_constants(test, cols)
		labels, _ = remove_constants(labels, cols)
		assert train.shape[1] == test.shape[1] == labels.shape[1]
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		print("processed train shape:", train.shape)
		print("processed test shape:", test.shape)
		print("processed anomaly ratio:", labels.mean())
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, cols = remove_constants(train)
			test, _ = remove_constants(test, cols)
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			assert train.shape[1] == test.shape[1] == labels.shape[1]
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'WADI':
		dataset_folder = 'data/WADI'
		print("Raw CSV rows:", sum(1 for _ in open(os.path.join(dataset_folder, 'test_data_cleaned.csv'))))
		train = pd.read_csv(os.path.join(dataset_folder, 'train_data_cleaned.csv'), skiprows=1000, nrows=4e5)
		test_orig = pd.read_csv(os.path.join(dataset_folder, 'test_data_cleaned.csv'))
		print("Train rows after skipping 1000:", train.shape[0])
		train.dropna(how='all', inplace=True); test_orig.dropna(how='all', inplace=True)

		train.fillna(0, inplace=True); test_orig.fillna(0, inplace=True)

		test = test_orig[test_orig.columns[:-1]].copy()
		labels = np.zeros(test.shape)
		for i in range(test.shape[0]):
			if test_orig[test_orig.columns[-1]][i] == -1: # for WADI we use, 1 is normal, -1 is attack
				labels[i, :] = 1 ## mark the whole row as anomalous if it is an attack
		train, test, labels = train.values[::5, :].astype(float), test.values[::5, :].astype(float), labels[::5, :].astype(float)
		train, cols = remove_constants(train)
		print("Train rows after preprocessed:", train.shape[0])
		test, _ = remove_constants(test, cols)
		labels, _ = remove_constants(labels, cols)
		assert train.shape[1] == test.shape[1] == labels.shape[1]
		train = (train - train.min(0)) / (train.ptp(0) + 1e-4)
		test = (test - test.min(0)) / (test.ptp(0) + 1e-4)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'MBA':
		dataset_folder = 'data/MBA'
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		ls = ls.values[:,1].astype(int)
		labels = np.zeros_like(test)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")