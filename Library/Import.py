###############################################################################
# This file import all necessary general python libraries 
# Author: Jean-loup Faulon jfaulon@gmail.com
# Updates: 24/11/2023, 20/02/2025
###############################################################################

from __future__ import print_function
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import csv
import random
import math
import numpy as np
import pandas as pd
import time
import json
import copy
import pickle
import sklearn
import xgboost
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # no warnings
import tensorflow as tf
import keras
import keras.backend as K
from keras import initializers
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'
global_training_temperature = 10 # For training with temperature (Gumbel-softmax)

