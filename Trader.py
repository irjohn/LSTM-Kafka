# IMPORTS

from client.client import PhemexClient
from database import Database
from datetime import datetime
from consumer import Consumer
import pandas as pd
import signal
#----------------------------------------------------------------------------------------------->



class Trader:
	streams = ['sBTCUSDT','sETHUSDT','sADAUSDT','sSOLUSDT']
	def __init__(self,**kwargs):
		signal.signal(signal.SIGINT,self.handler)
		self.client = PhemexClient()
		self.calc_fees = lambda qty,price,rate: (((qty*1)/price)*rate)*price
		self.long_pnl = lambda qty,avg_entry,exit: (((qty*1)/avg_entry)-((qty*1)/exit))-self.calc_fees(qty,exit,self.taker_fee_rate)
		self.short_pnl = lambda qty,avg_entry,exit: (((qty*1)/exit)-((qty*1)/avg_entry))-self.calc_fees(qty,exit,self.taker_fee_rate)
		self.maker_fee_rate = -0.00025
		self.taker_fee_rate = 0.00075
		self.Ep = 10000
		self.Ev = 100000000
		
		## TEST ##
		self.init_balance = 100
		self.contract_cost = 1
		## END TEST ##
		for stream in Trader.streams:
			if self.client.asset in stream:
				self.stream = stream


		# STREAM
		self.res = kwargs.get('res','h')

		## TEST ##
		self.stream = 'sETHUSDT'
		self.res = 'm'
		## END TEST ##

		self.servers = kwargs.get('servers',['192.168.0.156:9092'])
		self.streamer = Consumer(self.servers,self.stream,self.res,consumer_type='trader')
		self.trade_history = pd.DataFrame(columns=['ts','prediction','open','close','qty'])

	def handler(self,signal,frame):
		from sys import exit
		print('Saving trade history...')
		self.trade_history.to_csv('trades.csv',index=False)
		exit(0)

	def get_long_pnl(self,reduce_qty,entry_price,exit_price):
		open_value = (reduce_qty*1)/entry_price
		close_value = (reduce_qty*1)/exit_price
		pnl = (open_value-close_value)*exit_price
		fee = self.calc_fees(reduce_qty,exit_price,self.taker_fee_rate)
		print(f'sell {reduce_qty} | price {exit_price}')
		print(f'open value {open_value}')
		print(f'close value {close_value}')
		print(f'entry price {entry_price}')
		print(f'exit price {exit_price}')
		print(f'pnl {pnl}')
		print(f'fee {fee*2}')
		print(f'total pnl {pnl-(fee*2)}')
		print('----------'*8)
		pnl -= fee
		self.pnls.append(pnl-fee)
		return pnl

	def get_short_pnl(self,reduce_qty,entry_price,exit_price):
		open_value = (reduce_qty*1)/entry_price
		close_value = (reduce_qty*1)/exit_price
		pnl = (close_value-open_value)*exit_price
		fee = self.calc_fees(reduce_qty,exit_price,self.taker_fee_rate)
		print(f'buy {reduce_qty} | price {exit_price}')
		print(f'open value {open_value}')
		print(f'close value {close_value}')
		print(f'entry price {entry_price}')
		print(f'exit price {exit_price}')
		print(f'pnl {pnl}')
		print(f'fee {fee*2}')
		print(f'total pnl {pnl-(fee*2)}')
		print('----------'*8)
		pnl -= fee
		self.pnls.append(pnl-fee)
		return pnl

	def make_transaction(self,qty,entry,exit):
		trade = pd.DataFrame({'ts':self.ts,
							  'prediction':self.pred,
							  'open':self.entry_price,
							  'close':self.last_close},
							  columns=['ts','prediction','open','close','qty'])
		self.trade_history = pd.concat([self.trade_history,trade])


	def ingest(self):
		for data in self.streamer.consumer:
			data = data.value
			ts,pred,open_price,last_close = (datetime.fromtimestamp(data[0]),data[1],data[2],data[3])
			order =	{'symbol': self.settings.symbol,
					 'clOrdID': f'{price}-{int(time.time()*1000)}',
					 'side': 'Buy',
					 'orderQty': self.getQty(price),
					 'priceEp': price*self.Ep,
					 'ordType': 'Market',
					 'timeInForce': 'GoodTillCancel'}



	def test_ingest(self):
		self.long_position = False
		self.short_position = False
		self.balance = self.init_balance
		self.pnls = []
		seq = 0
		pnl = 0
		print('preparing to ingest...')
		for data in self.streamer.consumer:
			data = data.value
			self.ts,self.pred,self.open_price,self.last_close = (datetime.fromtimestamp(data[0]),data[1],data[2],data[3])
			print(f'Prediction for: {self.ts} | Prediction: {self.pred} | Open: {self.open_price} | Last close: {self.last_close}')
			if self.pnls:
				print(f'Balance: {self.balance} | Last pnl: {self.pnls[-1]}')
			else:
				print(f'Balance: {self.balance} | Last pnl: {pnl}')
			if self.long_position:
				pnl = self.get_long_pnl(self.qty,self.entry_price,self.last_close)
				self.balance += pnl
				self.position = False
				self.qty = 0
			elif self.short_position:
				pnl = self.get_short_pnl(self.qty,self.entry_price,self.last_close)
				self.balance += pnl
				self.position = False
				self.qty = 0				
			if self.pred == 1:
				self.entry_price = self.open_price
				self.qty = int(self.balance/self.contract_cost)
				print(f'long {self.qty} | price {self.entry_price}')
				self.balance -= self.calc_fees(self.qty,self.entry_price,self.taker_fee_rate)
				self.long_position = True
			elif self.pred == 0:
				self.entry_price = self.open_price
				self.qty = int(self.balance/self.contract_cost)
				print(f'short {self.qty} | price {self.entry_price}')
				self.balance -= self.calc_fees(self.qty,self.entry_price,self.taker_fee_rate)
				self.short_position = True
			print('----------'*8)		
			seq += 1





if __name__ == '__main__':
	trader = Trader()
	trader.test_ingest()