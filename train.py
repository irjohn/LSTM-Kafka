#!/usr/bin/env python3

from database import Database
from funcs import sort_features,scale,train
import time
#--------------------------------------------------------------------------------------------------->
'''
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
	'willr':True,
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
'''
#--------------------------------------------------------------------------------------------------->
# FEATURES
feature_dct = { 'target': ['move'],
				'scaled':{
					'minmax': [],
					'standard': ['trima12','trima26','natr'],
					'maxabs': []
				},
				'non_scaled': ['bop','bbp'],
}
feature_dct['all'] = feature_dct['non_scaled']+feature_dct['scaled']['minmax']+feature_dct['scaled']['maxabs']+feature_dct['scaled']['standard']
n_features = len(feature_dct['all'])
feature_dct,indicators = sort_features(feature_dct) # Use kwargs to override timeperiods


# DATA
symbol = 'sBTCUSDT'
resolution = 'd'
with Database() as db:
	df = db.getDF(symbol,setindex=True,resolution=resolution,labels=False,**indicators).iloc[:-1]


# SCALE
df,scales = scale(df,feature_dct)
print(df.tail())


# PARAMS
Time = int(time.time())
filename = f'{symbol}-{Time}-{resolution}'
params = dict(
			name=symbol,
			plot=False,
			filename=filename,
			time=Time,
			maximize='val_fbeta',
			feature_dct=feature_dct,
			features=feature_dct['all'],
			target=feature_dct['target'],
			impatient=True,
			epochs=2,
			batch_size=2,
			n_forecast=1,
			n_lookback=24,
			n_features=n_features,
			n_neurons=200,
			dropout=0.05,
			build_once=True,
)

train(params,df,threshold=0.9,save=True)


