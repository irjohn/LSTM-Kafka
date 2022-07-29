#!/usr/bin/env python3

import json
from os.path import join
#--------------------------------------------------------------->
path = join('client','config.json')

class Config:
	def __init__(self):
		with open(path,'r') as f:
			config = json.load(f)
		self.id = config['id']
		self.secret = config['secret']
		self.asset = config['asset'] # Example: BTC,ETH
		self.symbol = config['symbol'] # Example: BTCUSD,ETHUSD,
		self.email = config['email']
		self.email_password = config['email_password']
		self.text = config['text']

