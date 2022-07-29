import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import numpy as np
#----------------------------------------------------------------------------------->
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

class EarlyStopping(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0,timer=False,target=('val_fbeta',0.9),impatient=False):
        super(EarlyStopping, self).__init__()
        self.impatient = impatient
        self.target_name = target[0].lower()
        self.target_value = target[1]
        # if streaming, limit training time
        if timer:
            if datetime.now().minute > 50:
                self.stopping_time = datetime.now().replace(hour=datetime.now().hour+1,minute=50,second=0,microsecond=0)
            else:
                self.stopping_time = datetime.now().replace(minute=50,second=0,microsecond=0)
        else:
            self.stopping_time = False
        # set patience to infinity if no value
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # The best epoch
        self.best_epoch = 0
        if self.target_name == 'loss' or self.target_name == 'val_loss':
            # Initialize the best as infinity.
            self.best = np.Inf
        else:
            # Initialize the best as 0
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.target_name)
        if self.target_name == 'loss' or self.target_name == 'val_loss':
            if np.less(current, self.best):
                self.best = current
                self.wait = 0
                # Record the best weights if current results is better (less).
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print("Restoring model weights from the end of the best epoch.")
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
        else:
            if np.greater(current, self.best):
                self.best = current
                self.best_epoch = epoch+1
                self.wait = 0
                # Record the best weights if current results is better (greater).
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print("Restoring model weights from the end of the best epoch.")
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
        if self.stopping_time:
            if datetime.now() >= self.stopping_time:
                print(self.stopping_time)
                print("Time limit reached...")
                print("Restoring model weights from the end of the best epoch.")
                self.stopped_epoch = epoch
                self.model.history.history['early_stop'] = True
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
        if self.impatient:
            if current > self.target_value:
                print("Restoring model weights from the end of the best epoch.")
                print("Impatience set and current exceeds threshold")
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        self.model.set_weights(self.best_weights)
        print(f'Best epoch: {self.best_epoch} | Best {self.target_name}: {self.best}') 
        print("Restoring model weights from the end of the best epoch.")
