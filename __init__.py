from blocks import ConvNormPool
from blocks import ConvNormPool
from blocks import DepthwiseSperableConv

from models import DenseNet

import sys

if __package__ is not None:
    sys.path.insert(1,sys.path[0]+'/'+__package__)

