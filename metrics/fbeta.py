# IMPORTS
import tensorflow as tf
from tensorflow.keras.metrics import Metric
#--------------------------------------------------------------------------------------------------->
class FBeta(Metric):
  def __init__(self, name='fbeta', beta=1, threshold=0.5, epsilon=1e-7, **kwargs):
    # initializing an object of the super class
    super(FBeta, self).__init__(name=name, **kwargs)

    # initializing state variables
    self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives 
    self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
    self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives

    # initializing other atrributes that wouldn't be changed for every object of this class
    self.beta_squared = beta**2 
    self.threshold = threshold
    self.epsilon = epsilon

  def update_state(self, ytrue, ypred, sample_weight=None):
    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.float32)
    ypred = tf.cast(ypred, tf.float32)

    # setting values of ypred greater than the set threshold to 1 while those lesser to 0
    ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
        
    self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives atrribute
    self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
    self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive atrribute

  def result(self):
    self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
    self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall

    # calculating fbeta
    self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)
    
    return self.fb

  def reset_state(self):
    self.tp.assign(0) # resets true positives to zero
    self.predicted_positive.assign(0) # resets predicted positives to zero
    self.actual_positive.assign(0) # resets actual positives to zero