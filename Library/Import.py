###############################################################################
# This file import all necessary general python libraries 
# Author: Jean-loup Faulon jfaulon@gmail.com
# Updates: 24/11/2023
###############################################################################

from __future__ import print_function
import os
import sys
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
import tensorflow as tf
#import tensorflow_probability as tfp
import keras
import keras.backend as K
from keras import initializers
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no warnings
global_training_temperature = 10 # For training with temperature (Gumbel-softmax)

