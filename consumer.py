from kafka import KafkaConsumer
from kafka import TopicPartition
from json import loads
import sys
import asyncio

class Consumer:
	deserializer = lambda x: loads(x.decode('utf-8'))
	auto_offset = 'earliest'
	def __init__(self,servers,topicName,resolution='m',consumer_type='stream'):
		group_id = topicName
		if consumer_type.lower() == 'stream':
			self.resolutions = {'m':0,'h':1,'d':2}
		elif consumer_type.lower() == 'trader':
			self.resolutions = {'m':3,'h':4,'d':5}
		else:
			raise ValueError('consumer_type must be either "stream" or "trader"')
		if resolution[0].lower() in self.resolutions.keys():
			self.partition = self.resolutions[resolution[0].lower()]
		self.stream = topicName
		self.consumer = KafkaConsumer(
		     auto_offset_reset='latest',
			 enable_auto_commit=True,
			 api_version=(3,2,0),
             bootstrap_servers=servers,
             value_deserializer=lambda x: loads(x.decode('utf-8')))
		self.consumer.assign([TopicPartition(f'{topicName}', self.partition)])

#------------------------------------------------------------------------
#if __name__ == '__main__':
#    consumer = Consumer(['kali:9092'],'sETHUSDT',group_id='sETHUSDT')
#    for message in consumer.consumer:
#        print(message)