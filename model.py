from keras.models import load_model
from database import Database
from layers.attention import Attention
from metrics.fbeta import FBeta
from funcs import find_model,save_model,clean_models,scale
import numpy as np
import pandas as pd
import os
#---------------------------------------------------------------->

# PARAMS
symbol = 'sETHUSDT'
res = 'h'
filename = find_model(symbol,res)

# MODEL
model = load_model(os.path.join('models',filename),custom_objects={'Attention':Attention,'FBeta':FBeta})


