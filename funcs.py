#!/usr/bin/env python3
#----------------------------------------------------------------------->
# IMPORTS

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.metrics import Accuracy,Precision,Recall,AUC
from metrics.fbeta import FBeta
from callbacks.callbacks import EarlyStopping
from layers.attention import Attention
from datetime import datetime
from decorators import timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from platform import platform
import os
#----------------------------------------------------------------------->

def get_featuresets(features):
	import random
	import itertools
	random.shuffle(features)
	feature_sets = []
	for L in range(0, len(features)+1):
		for subset in itertools.combinations(features, L):
			if len(subset) > 0:
				subset = list(subset)
				feature_sets.append(subset)
	return feature_sets

def get_feature_generator(features):
	import random
	import itertools
	random.shuffle(features)
	for L in range(0, len(features)+1):
		for subset in itertools.combinations(features, L):
			if len(subset) > 0:
				yield list(subset)	



def scale(df,features):
	standard = features['scaled'].get('standard',[])
	minmax = features['scaled'].get('minmax',[])
	maxabs = features['scaled'].get('maxabs',[])
	scalers = {'standard':None,'minmax':None,'maxabs':None}
	if 'diff' in features['indicators']:
		minmax.append('diff')

	if standard:
		from sklearn.preprocessing import StandardScaler
		standardScaler = StandardScaler().fit(df[features['scaled']['standard']])
		df[features['scaled']['standard']] = standardScaler.transform(df[features['scaled']['standard']])
		scalers['standard'] = standardScaler
		
	if minmax:
		from sklearn.preprocessing import MinMaxScaler
		minmaxScaler = MinMaxScaler().fit(df[features['scaled']['minmax']])
		df[features['scaled']['minmax']] = minmaxScaler.transform(df[features['scaled']['minmax']])
		scalers['minmax'] = minmaxScaler

	if maxabs:
		from sklearn.preprocessing import MaxAbsScaler
		maxabsScaler = MaxAbsScaler().fit(df[features['scaled']['maxabs']])
		df[features['scaled']['maxabs']] = maxabsScaler.transform(df[features['scaled']['maxabs']])
		scalers['maxabs'] = maxabsScaler
	return df,scalers


def get_model_attributes(symbol,filename,data_type='df',**kwargs):
	from json import load
	from database import Database
	path = os.path.join('models',filename,'model_info.json')
	try:
		with open(path,'r') as f:
			models = load(f)
	except FileNotFoundError:
		raise FileNotFoundError('No model information file found. Make sure it is located in current directory.')

	if data_type.lower() == 'all' or 'param' in data_type.lower():
		return models[filename]['parameters']

	indicators = models[filename]['parameters']['feature_dct']['indicators']
	feature_dct = models[filename]['parameters']['feature_dct']
	for k,v in indicators.items():
		if v == 'true':
			indicators[k] = True

	with Database() as db:
		setindex = kwargs.get('setindex',True)
		labels = kwargs.get('labels',False)
		df = db.getDF(symbol,setindex=setindex,labels=labels,resolution=filename[-1],**indicators)
	return df,feature_dct


def clean_models():
	from json import load
	from os import remove
	from os.path import join
	print('Cleaning models directory...')
	try:
		with open('model_info.json','r') as f:
			models = load(f)
	except FileNotFoundError:
		raise('No model information file found. Make sure it is located in current directory.')

	for file in os.listdir('models'):
		if file not in models:
			print(f'{file} is not in models!')
			print('Removing...')
			remove(join('models',file))


def build_model(*args,**kwargs):
	custom_metrics = kwargs.get('custom_metrics',[])
	n_forecast = kwargs.get('n_forecast',1)
	n_lookback = kwargs.get('n_lookback',24)
	n_neurons = kwargs.get('n_neurons',200)
	n_features = kwargs.get('n_features',len(kwargs['features']))
	dropout = kwargs.get('dropout',0.05)
	loss = kwargs.get('loss','binary_crossentropy')
	optimizer = kwargs.get('optimizer','adam')
	shape = (n_lookback,n_features)


	model = Sequential()
	model.add(LSTM(units=n_neurons,return_sequences=True,input_shape=shape))
	model.add(Attention(return_sequences=False))
	model.add(Dropout(dropout))
	model.add(Dense(1440,activation='sigmoid'))
	model.add(Dense(n_forecast,activation='sigmoid'))
	model.compile(optimizer=optimizer,loss=loss,metrics=['Accuracy','Precision','Recall','AUC']+custom_metrics)
	kwargs['model_data'] = {'layers': [layer.get_config() for layer in model.layers],
							'optimizer':optimizer,
							'loss':loss}
	return model,kwargs

def find_model(symbol,res=None):
	import re
	import os
	directory = os.listdir('./models')
	matches = []
	if res:
		regex = re.compile(f'[0-9]+-{res.lower()}')
		for file in directory:
			if symbol in file:
				match = regex.findall(file)
				if len(match) > 0:
					matches.append(match[0][:-2])
		matches = list(map(lambda x: int(x),matches))
		if len(matches) > 0:
			match = max(matches)
			return f'{symbol}-{match}-{res}'
		else:
			return 'No models'		
	else:
		regex = re.compile('[0-9]+.')	
		for file in directory:
			if symbol in file:
				match = regex.findall(file)
				if len(match) > 0:
					matches.append(match[0][:-1])
		matches = list(map(lambda x: int(x),matches))
		if len(matches) > 0:
			match = max(matches)
			return f'{symbol}-{match}.h5'
		else:
			return 'No models'

def save_model(params,filename):
	import json
	if 'custom_metrics' in params:
		del params['custom_metrics']
	path = os.path.join('models',filename,'model_info.json')
	output = {filename:{'model':params['model_data'],
						'parameters':params}}
	del output[filename]['parameters']['model_data']
	
	try:
		with open(path,'r') as f:
		    data = json.load(f)

		key = list(output.keys())[0]
		data[key] = output[key]

		with open(path,'w') as f:
			data = json.dump(data,f)
	except FileNotFoundError as e:
		print(e)
		with open(path,'w') as f:
			json.dump(output,f)


@timer
def train(params:dict,df,threshold=0.7,save=True):
	from tensorflow.keras.preprocessing import timeseries_dataset_from_array as TimeseriesGenerator
	"""Function to train an existing model or build a new one from parameter set

		Arguments:
		plot - 
	"""
	print(params)
	split = int(len(df)*2/3)
	
	# PARAMS
	plot = params.get('plot',False)
	filename = params.get('filename','model.h5')
	maximize = params.get('maximize','val_fbeta')
	patience = params.get('patience',np.Inf)
	impatient = params.get('impatient',False)
	max_tries = params.get('max_tries',np.Inf)
	n_epochs = params.get('epochs',20)
	batch_size = params.get('batch_size',2)
	n_lookback = params.get('n_lookback',24)
	n_forecast = params.get('n_forecast',1)
	timed = params.get('timed',False)
	impatient = params.get('impatient',False)
	increment = params.get('increment',5)
	train_type = params.get('train_type','train')
	model = params.get('model',None)
	build_once = False if model else params.get('build_once',False)
	recompile = False if model else False if build_once else params.get('recompile',True)


	if not model:
		if not recompile and not build_once:
			from sys import exit
			raise ValueError("No model was provided but compile is turned off. Please provide a model")
			exit(1)
	
	# DATA
	train_gen = TimeseriesGenerator(df[params['features']],
									df[params['target']],
									sequence_length=n_lookback,
									batch_size=batch_size,
									end_index=split)
	test_gen = TimeseriesGenerator(df[params['features']],
									df[params['target']],
									sequence_length=n_lookback,
									batch_size=batch_size,
									start_index=split)

	# BUILD
	if build_once or recompile:
		print('Compiling...')
		model,params = build_model(custom_metrics=[FBeta(beta=0.5,threshold=0.6)],**params)

	# FIT
	history = model.fit(train_gen,
					    epochs=n_epochs,
					    batch_size=batch_size,
					    validation_data=test_gen,
					    verbose = 0 if train_type == 'stream' else 1,
					    callbacks=[EarlyStopping(patience=patience,
										         target=(maximize,threshold),
											     timer=timed,
											     impatient=impatient)])
	value = max(history.history[maximize])
	max_tries -= 1
	while value < threshold and max_tries > 0:
		# REFIT
		n_epochs += increment
		if recompile:
			print('Recompiling...')
			model,params = build_model(custom_metrics=[FBeta(beta=0.5,threshold=0.6)],**params)
		history = model.fit(train_gen,
							epochs=n_epochs,
							batch_size=batch_size,
							validation_data=test_gen,
							verbose = 0 if train_type == 'stream' else 1,
							callbacks=[EarlyStopping(patience=patience,
													 target=(maximize,threshold),
													 timer=timed,
													 impatient=impatient)])
		value = max(history.history[maximize])
		max_tries -= 1
		if history.history.get('early_stop',None) == True:
			break

	# SAVE
	if save:
		params['epochs'] = n_epochs
		model.save(os.path.join('models',filename))
		try:
			save_model(params,filename)
		except Exception as e:
			print(e)
			pass

	if plot:
		plot_history(history)
	print(f'Returning model at: {datetime.now()}')
	return model

def train_combos(params:dict,df):
	pass

def sort_features(features,**override):
	exclude = ['open','high','low','close','volume']
	indicators = {k:True for k in features['all'] if k not in exclude}
	for k,v in override.items():
		indicators[k] = v
	features['indicators'] = indicators
	return features,indicators

def plot_history(history,both=False,seperate=True):
	if both:
		for key in history.history.keys():
			if 'val' not in key:
				plt.plot(history.history[key])
				plt.plot(history.history[f'val_{key}'])
				plt.xlabel('Epochs')
				plt.title(key)
				plt.ylabel(f'{key}')
				plt.legend(['train','test'],loc='upper left')
				plt.show()
		for key in history.history.keys():
			if 'val' in key:
				plt.plot(history.history[key])
		plt.xlabel('Epochs')
		plt.title('Metrics')
		plt.ylabel('Metrics')
		plt.legend([x for x in history.history.keys() if 'val' in x],bbox_to_anchor=(0.65, 1.25))
		plt.show()
	else:
		if seperate:
			for key in history.history.keys():
				if 'val' not in key:
					plt.plot(history.history[key])
					plt.plot(history.history[f'val_{key}'])
					plt.xlabel('Epochs')
					plt.title(key)
					plt.ylabel(f'{key}')
					plt.legend(['train','test'],loc='upper left')
					plt.show()
		else:
			for key in history.history.keys():
				if 'val' in key:
					plt.plot(history.history[key])
			plt.title('Metrics')
			plt.xlabel('Epochs')
			plt.ylabel('Metrics')
			plt.legend([x for x in history.history.keys() if 'val' in x],bbox_to_anchor=(0.65, 1.25))
			plt.show()


def check_os(os='Windows'):
	if os.title() in platform():
		return True
	else:
		return False

def streamer_usage():
	from sys import exit
	print('Usage:')
	print('streamer.py [OPTIONS]')
	print('* denotes required argument')
	print('-h  | --help                 | print out this usage prompt')
	print('-b  | --batch-size           | specify a batch size')
	print('-B  | --build-once           | specify whether to recompile once')
	print('-f  | --filename             | specify a specific model in the models directory')
	print('-i  | --increment            | specify epoch increments between trains')
	print('-I  | --impatient            | specify whether to train imaptiently i.e. stop training if loss is above threshold')
	print('-l  | --log                  | specify a logpath')
	print('-m  | --max-tries            | specify whether to limit number of trains')
	print('-M  | --maximize             | specify a metric to maximize/minimize')
	print('-p  | --plot                 | specify whether to plot training history')
	print('-R  | --recompile            | specify whether to recompile new model everytime')
	print('*-s | *--symbol              | symbol to stream. accepts one of the following: [btc,eth,ada,sol]')
	print('*-S | *--servers             | comma separated Kafka servers in the format <ip/host>:<port>')
	print('-t  | --train | --no-train   | specify whether to train new models. -t accepts: [true/t/0/false/f/1]')
	print('-T  | --timed                | specify whether to cut the training short at the 0:50 minute mark')
	print('-r  | --resolution           | one of the following: [m,h,d] defaults to "h"')
	exit(1)
#----------------------------------------------------------------------------->
# DATA


if __name__ == "__main__":
	pass