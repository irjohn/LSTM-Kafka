#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:15:14 2022

@author: john
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters as hp
from keras_tuner.tuners import BayesianOptimization,RandomSearch
from keras_tuner import Objective
from keras.metrics import MeanAbsoluteError,RootMeanSquaredError,MeanAbsolutePercentageError,MeanSquaredError
from attention import Attention
from tensorflow.keras.callbacks import EarlyStopping
from db.database import Database


def model(hp):
	hp_attention_begin = hp.Boolean('attention begin')
	hp_attention_end = hp.Boolean('attention end')

	hp_units1 = hp.Int('units1', min_value=5, max_value=200, step=5)
	hp_units2 = hp.Int('units2', min_value=5, max_value=200, step=5)
	hp_drop1 = hp.Float('hp_drop1',min_value=0.01,max_value=0.2, step=0.01)
	hp_drop2 = hp.Float('hp_drop2',min_value=0.01,max_value=0.2, step=0.01)

	# create model
	model = Sequential()
	model.add(LSTM(units=200,return_sequences=False,input_shape=shape))
	model.add(Dense(1440,activation='sigmoid'))
	model.add(Dropout(0.05))
	model.add(Dense(n_forecast,activation='sigmoid'))
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['Accuracy','Precision','Recall','AUC']+custom_metrics)
	return model

class RandomTuner(RandomSearch):
	def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
		kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 1, 10, step=1)
		#kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 1000, step=50)
		print(f'"batch_size": {kwargs["batch_size"]}, "epochs": {kwargs["epochs"]}')
		return super(RandomTuner, self).run_trial(trial, *args, **kwargs)# Uses same arguments as the RandomSearch Tuner.


class BayesianTuner(BayesianOptimization):
	def __init__(self,df):
		self.df = df
		shape = (args['n_lookback'],n_features)
		model = build_model(shape,metrics=[FBeta(beta=0.5)],**args)
		train_gen = TimeseriesGenerator(df[features['all']],
										df[target],
										sequence_length=args['n_lookback'],
										batch_size=args['batch_size'],
										end_index=split)
		test_gen = TimeseriesGenerator(df[features['all']],
										df[target],
										sequence_length=args['n_lookback'],
										batch_size=args['batch_size'],
										start_index=split)

			
	def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
		kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 1, 10, step=1)
		#kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 1000, step=50)
		print(f'"batch_size": {kwargs["batch_size"]}, "epochs": {kwargs["epochs"]}')
		return super(BayesianTuner, self).run_trial(trial, *args, **kwargs)# Uses same arguments as the BayesianOptimization Tuner.


#--------------------------------------------------------------------------------------------------->
indicators = {
	'sma':True,
	'ema':True,
	'wma':True,
	'vma':True,
	'trima':True,
	'rsi':True,
	'uo':True,
	'roc':True,
	'rocp':True,
	'kst':True,
	'bbands':True,
	'bbp':True,
	'atr':True,
	'natr':True,
	'trange':True,
	'apo':True,
	'obv':True,
	'adosc':True,
	'trend':True,
	'ppo':True,
	'adx':True,
	'adxr':True,
	'bop':True,
	'cci':True,
	'mfi':True,
	'stddev':True,
	'var':True,
}
#--------------------------------------------------------------------------------------------------->
symbol = 'sBTCUSDT'
res = 'h'

with Database() as db:
    df = db.getDF(symbol,**indicators)


#--------------------------------------------------------------------------------------------------->
#randomTuner = RandomTuner(model,
#				objective=Objective('val_fbeta','max'),
#				project_name=symbol,
#				max_trials=100000)
bayesianTuner = BayesianTuner(model,
				objective=Objective('val_fbeta','max'),
				project_name=symbol,
				max_trials=100000)

# Don't pass epochs or batch_size here, let the Tuner tune them.
# Will stop training if the "val_loss" hasn't improved in 3 epochs.
#Tuner.search(X_train, y_train,validation_data=(X_test,y_test), callbacks=[EarlyStopping('loss', patience=5)])

#bayesianTuner.search(X_train, y_train,epochs=10,validation_data=(X_test,y_test), callbacks=[EarlyStopping('val_mean_absolute_percentage_error', patience=3)])
bayesianTuner.search(X_train, y_train,epochs=10,validation_data=(X_test,y_test))