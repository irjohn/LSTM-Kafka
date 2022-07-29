#!/usr/bin/python3

import MySQLdb
import json
import time
from math import ceil
from termcolor import colored
import pandas as pd
import talib as ta
import numpy as np
import os
#---------------------------------------------------------------------------------------->

class Database():
	HOME = os.path.expanduser('~')
	def __init__(self,directory=False):
		if directory:
			self.HOME = directory
		else:
			self.HOME = Database.HOME
		with open(self.HOME + '/.database.json','r') as f:
			db_settings = json.load(f)
		self.conn = None
		self.cursor = None
		self.host = db_settings['host']
		self.username = db_settings['username']
		self.password = db_settings['password']
		self.db_name = db_settings['database']
		try:
			self.connect()
		except MySQLdb._exceptions.DatabaseError as e:
			print('[-] Connection error')
			print(e)
		except MySQLdb._exceptions.OperationalError as e:
			print('[-] Operational error')
			print(e)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_tb):
		if exc_type:
			print(exc_type,exc_value,exc_tb,sep="\n")
		self.close()

# Init
#----------------------------------- Connection ------------------------------------------>
# Connection


	def connect(self):
		# define the server and the database
		self.conn = MySQLdb.connect(host = self.host,
								    user = self.username,
									password = self.password,
									db = self.db_name)
		self.cursor = self.conn.cursor()
		return True

	def return_connection(self):
		# define the server and the database
		self.conn = MySQLdb.connect(host = self.host,
								    user = self.username,
									password = self.password,
									db = self.db_name)
		self.cursor = self.conn.cursor()
		return self.conn,self.cursor

	def close(self):
		self.conn.close()
		return True


# Connection
#---------------------------------------- Basic ------------------------------------------->
# Basic

	def round05(cls,n):
		return (ceil(n*20)/20)

	def query(self,query):
		self.cursor.execute(query)
		q = self.cursor.fetchall()
		self.conn.commit()
		return q

	def execute(self,query,data_tuple=None):
		if data_tuple:
			self.cursor.execute(query,data_tuple)
			self.conn.commit()
		else:
			self.cursor.execute(query)
			self.conn.commit()
		return True


# Basic
#---------------------------------------- Gets ------------------------------------------->
# Gets


	def getDF(self,symbol,csv=False,setindex=False,classify=False,target=False,labels=True,prioritize='buy',resolution='m',**kwargs):
		resolutions = ['m','h','d']
		if resolution[0].lower() in resolutions:
			resolution = resolutions[resolutions.index(resolution[0].lower())]
			table = f'crypto_1{resolution}_kline'
		else:
			raise ValueError(f'Must be one of the following: {resolutions}')
		q = f'SELECT * FROM {table} WHERE symbol = "{symbol}"'
		df = pd.read_sql_query(q,self.conn)
		df = df.iloc[:,1:]

		## CLASSIFICATION ##
		if classify:
			df['ON_returns'] = df['close'] - df['open'].shift(-1)
			df['ON_returns'] = df['ON_returns'].shift(1)
			df['ON_returns_signal'] = np.where(df['ON_returns']<0, 'up', 'down')
			df = pd.get_dummies(df, columns=['ON_returns_signal'])
		if not labels:
			if target:
				target = -target/100
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, 1, 0)
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, 0, 1)
			else:
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, 1, 0)
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, 0, 1)
		else:
			if target:
				target = -target/100
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, "Buy", "Sell")
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, "Sell", "Buy")
			else:
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, "Buy", "Sell")
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, "Sell", "Buy")			
		## END CLASSIFICATION ##

		df = self.get_indicators(df,kwargs)

		if csv:
			path = os.path.join('csv',str(symbol)+'.csv')
			df.to_csv(path,index=False)

		if setindex:
			df['ts'] = pd.to_datetime(df['ts'], format='%Y-%m-%d %H:%M:%S') 
			df.set_index('ts', inplace=True)
			return df.dropna()
		return df.dropna().reset_index(drop=True)



	def get_ether_df(self,resolution='d',csv=False,target=False,**kwargs):
		import yfinance as yf
		resolutions = ['m','h','d']
		if resolution[0].lower() == 'd':
			period = '10y'
			interval = '1d'
		elif resolution[0].lower() == 'h':
			interval = '1h'
			period = '730d'

		df = yf.download('ETH-USD',interval=interval,period=period)
		df.rename(columns={column:column.lower() for column in df.columns},inplace=True)
		df.index = df.index.tz_convert("America/Chicago")

		## CLASSIFICATION ##
		if target:
			target = -target/100
			df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, "Buy", "Sell")
		else:
			df['move'] = np.where(df['close']-df['close'].shift(-1)<0, "Buy", "Sell")
		## END CLASSIFICATION ##

		df = self.get_indicators(df,kwargs)
		if csv:
			path = os.path.join('csv',str(symbol)+'.csv')
			df.to_csv(path,index=False)
		return df.dropna()


	def get_stock_df(self,symbol,setindex=False,csv=False,target=False,labels=True,prioritize='buy',**kwargs):
		q = f'SELECT * FROM historical WHERE symbol = "{symbol}"'
		df = pd.read_sql_query(q,self.conn)

		## CLASSIFICATION ##
		df['ON_returns'] = df['close'] - df['open'].shift(-1)
		df['ON_returns'] = df['ON_returns'].shift(1)
		df['ON_returns_signal'] = np.where(df['ON_returns']<0, 'up', 'down')
		df = pd.get_dummies(df, columns=['ON_returns_signal'])
		if not labels:
			if target:
				target = -target/100
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, 1, 0)
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, 0, 1)
			else:
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, 1, 0)
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, 0, 1)
		else:
			if target:
				target = -target/100
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, "Buy", "Sell")
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<=df['close']*target, "Sell", "Buy")
			else:
				if prioritize.lower() == 'buy':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, "Buy", "Sell")
				elif prioritize.lower() == 'sell':
					df['move'] = np.where(df['close']-df['close'].shift(-1)<0, "Sell", "Buy")	
		## END CLASSIFICATION ##

		df = self.get_indicators(df,kwargs)

		if csv:
			path = os.path.join('csv',str(symbol)+'.csv')
			df.to_csv(path,index=False)

		if setindex:
			df['date'] = pd.to_datetime(df['date'])
			df.set_index('date', inplace=True)

		df = df.drop('ID',axis=1)
		return df.dropna()

	def get_indicators(self,df,kwargs):
		self.moving_averages = ['sma','ema','wma','tma','trima']
		for key,value in kwargs.items():
			key = key.lower()
			if key == 'diff':
				df = self.compute_differences(df,key,value)

			if key == 'stddev':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}	
				stddev = ta.STDDEV(df.close, timeperiod=kwargs[key].get('timeperiod',5), nbdev=kwargs[key].get('nbdev',2))
				df = df.assign(stddev=stddev)

			if key == 'var':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}
				var = ta.VAR(df.close, timeperiod=kwargs[key].get('timeperiod',5), nbdev=kwargs[key].get('nbdev',2))
				df = df.assign(var=var)

#--------------------------------------------------------------------------------------------------------------------------->
# MOVING AVERAGES 

			if key == 'sma':
				if type(kwargs[key]) == bool:
					kwargs[key] = 20
				if type(kwargs[key]) == list:
					for i in range(len(kwargs[key])):
						df[f'{key}{kwargs[key][i]}'] = ta.SMA(df.close,timeperiod=kwargs[key][i])
				else:	
					sma = ta.SMA(df.close,timeperiod=kwargs[key])
					df = df.assign(sma=sma)
			elif 'sma' in key:
				if len(key) > 3:
					period = int(key[3:])
				sma = ta.SMA(df.close,timeperiod=period)
				df[f'{key}'] = sma

			if key == 'ema':
				if type(kwargs[key]) == bool:
					kwargs[key] = 20
				if type(kwargs[key]) == list:
					for i in range(len(kwargs[key])):
						df[f'{key}{kwargs[key][i]}'] = ta.EMA(df.close,timeperiod=kwargs[key][i])
				else:
					ema = ta.EMA(df.close,timeperiod=kwargs[key])
					df = df.assign(ema=ema)
			elif 'ema' in key:
				if len(key) > 3:
					period = int(key[3:])
				ema = ta.EMA(df.close,timeperiod=period)
				df[f'{key}'] = ema

			if key == 'wma':
				if type(kwargs[key]) == bool:
					kwargs[key] = 20
				if type(kwargs[key]) == list:
					for i in range(len(kwargs[key])):
						df[f'{key}{kwargs[key][i]}'] = ta.WMA(df.close,timeperiod=kwargs[key][i])
				else:
					wma = ta.WMA(df.close,timeperiod=kwargs[key])
					df = df.assign(wma=wma)
			elif 'wma' in key:
				if len(key) > 3:
					period = int(key[3:])
				wma = ta.WMA(df.close,timeperiod=period)
				df[f'{key}'] = wma

			if key == 'trima':
				if type(kwargs[key]) == bool:
					kwargs[key] = 20
				if type(kwargs[key]) == list:
					for i in range(len(kwargs[key])):
						trima = ta.SMA(df.close,timeperiod=kwargs[key][i])
						trima = trima.rolling(kwargs[key][i]).mean()
						df[f'{key}{kwargs[key][i]}'] = trima
				else:
					trima = ta.SMA(df.close, timeperiod=kwargs[key])
					trima = trima.rolling(kwargs[key]).mean()
					df = df.assign(trima=trima)
			elif 'trima' in key:
				if len(key) > 5:
					period = int(key[5:])
				trima = ta.SMA(df.close, timeperiod=period)
				trima = trima.rolling(period).mean()
				df[f'{key}'] = trima

			if key == 'tma':
				if type(kwargs[key]) == bool:
					kwargs[key] = 20
				if type(kwargs[key]) == list:
					for i in range(len(kwargs[key])):
						df[f'{key}{kwargs[key][i]}'] = ta.TRIMA(df.close,timeperiod=kwargs[key][i])
				else:
					tma = ta.TRIMA(df.close, timeperiod=kwargs[key])
					df = df.assign(tma=tma)
			elif 'tma' in key:
				if len(key) > 3:
					period = int(key[3:])
				tma = ta.TRIMA(df.close,timeperiod=period)
				df[f'{key}'] = tma

			if key == 'vma':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}	
				vma = ta.MAVP(df.close.values,df.close.values,minperiod=kwargs[key].get('minperiod',20),
															  maxperiod=kwargs[key].get('maxperiod',200),
															  matype=0)
				df = df.assign(vma=vma)

			if key == 'rsi':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				rsi = ta.RSI(df.close,timeperiod=kwargs[key])
				df = df.assign(rsi=rsi)

			if key == 'roc':
				if type(kwargs['roc']) == bool:
					kwargs['roc'] = 14
				roc = ta.ROC(df.close,timeperiod=kwargs[key])
				df = df.assign(roc=roc)

			if key == 'rocp':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				rocp = ta.ROCP(df.close,timeperiod=kwargs[key])
				df = df.assign(rocp=rocp)

			if key == 'bbands':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}
				upper,middle,lower = ta.BBANDS(df.close,timeperiod=kwargs[key].get('timeperiod',14),
														nbdevup=kwargs[key].get('nbdevup',2),
														nbdevdn=kwargs[key].get('nbdevdown',2),
														matype=0)
				df = df.assign(lower=lower)
				df = df.assign(middle=middle)
				df = df.assign(upper=upper)

			if key == 'bbp':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}
				upper,middle,lower = ta.BBANDS(df.close,timeperiod=kwargs[key].get('timeperiod',14),
														nbdevup=kwargs[key].get('nbdevup',2),
														nbdevdn=kwargs[key].get('nbdevdown',2),
														matype=0)
				bbp = (df.close-lower)/(upper-lower)
				df = df.assign(bbp=bbp)

			if key == 'atr':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				atr = ta.ATR(df.high,df.low,df.close,timeperiod=kwargs[key])
				df = df.assign(atr=atr)

			if key == 'natr':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				natr = ta.NATR(df.high,df.low,df.close,timeperiod=kwargs[key])
				df = df.assign(natr=natr)

			if key == 'trange':
				trange = ta.TRANGE(df.high,df.low,df.close)
				df = df.assign(trange=trange)

			if key == 'apo':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}
				apo = ta.APO(df.close, fastperiod=kwargs[key].get('fastperiod',10),
									   slowperiod=kwargs[key].get('slowperiod',26),
									   matype=ta.MA_Type.EMA)
				df = df.assign(apo=apo)

			if key == 'ppo':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}
				ppo = ta.PPO(df.close, fastperiod=kwargs[key].get('fastperiod',26),
									   slowperiod=kwargs[key].get('slowperiod',10),
									   matype=ta.MA_Type.EMA)
				ppo_signal = ta.EMA(ppo,kwargs[key].get('signal',9))
				ppo_hist = ppo-ppo_signal
				df = df.assign(ppo=ppo,
							   ppo_signal=ppo_signal,
							   ppo_hist=ppo_hist)

			if key == 'obv':
				obv = ta.OBV(df.close,df.volume)
				df = df.assign(obv=obv)

			if key == 'adosc':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}	
				adosc = ta.ADOSC(df.high, df.low, df.close, df.volume, fastperiod=kwargs[key].get('fastperiod',3),
																	   slowperiod=kwargs[key].get('slowperiod',10))
				df = df.assign(adosc=adosc)

			if key == 'trend':
				trend = ta.HT_TRENDMODE(df.close)
				df = df.assign(trend=trend)

			if key == 'adx':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				adx = ta.ADX(df.high, df.low, df.close, timeperiod=kwargs[key])
				df = df.assign(adx=adx)

			if key == 'adxr':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				adxr = ta.ADXR(df.high, df.low, df.close, timeperiod=kwargs[key])
				df = df.assign(adxr=adxr)

			if key == 'bop':
				bop = ta.BOP(df.open, df.high, df.low, df.close)
				df = df.assign(bop=bop)

			if key == 'cci':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				cci = ta.CCI(df.high, df.low, df.close, timeperiod=kwargs[key])
				df = df.assign(cci=cci)

			if key == 'mfi':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				mfi = ta.MFI(df.high, df.low, df.close, df.volume, timeperiod=kwargs[key])
				df = df.assign(mfi=mfi)

			if key == 'uo':
				if type(kwargs[key]) == bool:
					kwargs[key] = {}
				uo = ta.ULTOSC(df.high, df.low, df.close, timeperiod1=kwargs[key].get('timeperiod1',7),
														  timeperiod2=kwargs[key].get('timeperiod2',14),
														  timeperiod3=kwargs[key].get('timeperiod3',28))
				df = df.assign(uo=uo)

			if key == 'willr':
				if type(kwargs[key]) == bool:
					kwargs[key] = 14
				willr = ta.WILLR(df.high,df.low,df.close,timeperiod=kwargs[key])
				df = df.assign(willr=willr)

			if key == 'kst':
				if type(kwargs[key]) == bool:
					kwargs[key] = dict(sma1=10,
										 sma2=10,
										 sma3=10,
										 sma4=15,
										 roc1=10,
										 roc2=15,
										 roc3=20,
										 roc4=30,
										 signal=9)

				df = self.get_kst(df,**kwargs[key])
		return df



	def get_kst(self,df,sma1=10, sma2=10, sma3=10, sma4=15, roc1=10, roc2=15, roc3=20, roc4=30, signal=9):
		rcma1 = ta.ROC(df.close,timeperiod=roc1).rolling(sma1).mean()
		rcma2 = ta.ROC(df.close,timeperiod=roc2).rolling(sma2).mean()
		rcma3 = ta.ROC(df.close,timeperiod=roc3).rolling(sma3).mean()
		rcma4 = ta.ROC(df.close,timeperiod=roc4).rolling(sma4).mean()
		kst = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
		kst_signal = kst.rolling(signal).mean()
		df = df.assign(kst=kst,
					   kst_signal=kst_signal)
		return df

	def compute_differences(self,df,key,lst):
		if type(lst) == list:
			moving_averages = ['sma','ema','wma','tma','trima']
			values = []
			min_period = np.Inf
			import re
			period_regex = re.compile('[0-9]+')
			name_regex = re.compile('[a-zA-Z]+')
			for item in lst:
				period = period_regex.findall(item)
				name = name_regex.findall(item)
				if name and period:
					name,period = name[0],int(period[0])
					if name not in self.moving_averages:
						raise ValueError("Must be a pair of moving averages to find the difference of in the form of '<name><period>' i.e. 'sma12'")
					if period < min_period:
						min_period = period
				if name == 'sma':
					value = ta.SMA(df.close,timeperiod=period)
				elif name == 'ema':
					value = ta.EMA(df.close,timeperiod=period)
				elif name == 'wma':
					value = ta.WMA(df.close,timeperiod=period)
				elif name == 'tma':
					value = ta.TRIMA(df.close,timeperiod=period)
				elif name == 'trima':
					value = ta.SMA(df.close,timeperiod=period)
					value = value.rolling(period).mean()
				container = [period,value]
				values.append(container)
			for lst in values:
				if lst[0] == min_period:
					idx = values.index(lst)
			if idx == 0:
				diff = values[0][1]-values[1][1]
			elif idx == 1:
				diff = values[1][1]-values[0][1]
			df['diff'] = diff
			return df				
		else:
			raise ValueError("Must be a pair of moving averages to find the difference of in the form of '<name><period>' i.e. 'sma12'")

#---------------------------------------- End ------------------------------------------->

if __name__ == '__main__':
	db = Database()