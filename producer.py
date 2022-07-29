from kafka import KafkaProducer
from json import dumps
#------------------------------------------------------------->
class Producer:
	serializer = lambda x: dumps(x).encode('utf-8')
	def __init__(self,servers,stream,resolution='h'):
		self.resolutions = {'m':3,'h':4,'d':5}
		if resolution[0].lower() in self.resolutions.keys():
			self.partition = self.resolutions[resolution[0].lower()]
		self.bootstrap_servers = servers
		self.Stream = stream
		self.producer = KafkaProducer(bootstrap_servers=servers,
						 			  api_version=(3,2,0),
						 			  value_serializer=Producer.serializer)

	def sendBundle(self,data,key=None,Stream=None):
		for entry in data:
			if Stream is not None:
				if key is not None:
					ack = self.producer.send(Stream,partition=key,value=entry)
				else:
					ack = self.producer.send(Stream,value=entry)
			else:
				if key is not None:
					ack = self.producer.send(self.Stream,partition=key,value=entry)
				else:
					ack = self.producer.send(self.Stream,value=entry)
			metadata = ack.get()
			print(f'Topic: {metadata.topic} :: Partition: {metadata.partition}')

	def send(self,data,key=None,Stream=None):
		if Stream is not None:
				if key is not None:
					ack = self.producer.send(Stream,partition=key,value=data)
				else:
					ack = self.producer.send(Stream,value=data)
		else:
			if key is not None:
				ack = self.producer.send(self.Stream,partition=key,value=data)
			else:
				ack = self.producer.send(self.Stream,value=data)
		metadata = ack.get()
		#print(f'Topic: {metadata.topic} :: Partition: {metadata.partition}')

	def send_prediction(self,data):
		ack = self.producer.send(self.Stream,partition=self.partition,value=data)
		metadata = ack.get()
		print(f'Topic: {metadata.topic} :: Partition: {metadata.partition}')