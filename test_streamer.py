# IMPORTS

from modelstreamer import ModelStreamer
from producer import Producer
from consumer import Consumer
import sys
import getopt
from funcs import streamer_usage
#---------------------------------------------------------------------->

symbol = False
servers = False
logpath = False
train = True
resolution = 'h'



argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'hl:s:S:t:r:', ['help','log','resolution=','symbol=','servers=','train','no-train'])
print(opts,args)
for o,a in opts:
	if o in ('-h','--help'):
		streamer_usage()

	elif o in ('-l','--log'):
		logpath = a

	elif o in ('-r','--resolution'):
		available = ['m','h','d']
		resolution = False
		for item in available:
			if a[0] in item.lower():
				resolution = item
		if not resolution:
			streamer_usage()
	elif o in ('-s','--symbol'):
		available = ['sBTCUSDT','sETHUSDT','sADAUSDT','sSOLUSDT']
		symbol = False
		for item in available:
			if a in item.lower():
				symbol = item
		if not symbol:
			streamer_usage()

	elif o in ('-S','--servers'):
		servers = a.split(',')

	elif o in ('-t','--train','--no-train'):
		if o == '--no-train':
			train = False
		elif o == '--train':
			train = True
		else:
			if a.lower() == 'false' or a.lower() == 'f' or a.lower() == 0:
				train = False
			elif a.lower() == 'true' or a.lower() == 't' or a.lower() == 1:
				train = True
			else:
				streamer_usage()

if not symbol or not servers:
	streamer_usage()
	sys.exit(1)


streamer = Consumer(servers,symbol,resolution=resolution)
producer = Producer(servers,symbol,resolution=resolution)

modelstreamer = ModelStreamer(symbol,
						      streamer,
							  resolution=resolution,
							  logpath=logpath if logpath else False,
							  producer=producer,
							  train_model=train
)
modelstreamer.ingest()






