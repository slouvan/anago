import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
import numpy as np
import random as rn
np.random.seed(1234)
rn.seed(1234)



#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

# Rest of code follows .:w
# Prepare training and validation data(steps, generatora
# print("INSIDE TRAIN function {}".format(x_train[0]) )
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K


tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from anago.reader import batch_iter
from keras.optimizers import Adagrad
# Train the model
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res


from keras.optimizers import Adam

from anago.metrics import get_callbacks
import anago.config as cfg

class Trainer(object):

    def __init__(self,
                 model,
                 training_config,
                 checkpoint_path=cfg.ANAGO_CHECKPOINT_DIR,
                 save_path='',
                 tensorboard=True,
                 preprocessor=None,
                 ):

        self.model = model
        self.training_config = training_config
        print("HELLO {}".format(cfg.ANAGO_CHECKPOINT_DIR))
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.preprocessor = preprocessor

    def train(self, x_train, y_train, x_valid=None, y_valid=None):

        train_steps, train_batches = batch_iter(x_train,
                                                y_train,
                                                self.training_config.batch_size,
                                                preprocessor=self.preprocessor)
        valid_steps, valid_batches = batch_iter(x_valid,
                                                y_valid,
                                                self.training_config.batch_size,
                                                preprocessor=self.preprocessor)


        self.model.compile(loss=self.model.crf.loss,
                           optimizer=Adam(lr=self.training_config.learning_rate),
                           )
        print("METRICS : {}".format(self.model.metrics_names))
        # Prepare callbacks

        print("Checkpoint : {}".format(self.checkpoint_path))
        callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                  tensorboard=self.tensorboard,
                                  eary_stopping=self.training_config.early_stopping,
                                  valid=(valid_steps, valid_batches, self.preprocessor))


        self.model.fit_generator(generator=train_batches,
                                 steps_per_epoch=train_steps,
                                 epochs=self.training_config.max_epoch,
                                 callbacks=callbacks)
