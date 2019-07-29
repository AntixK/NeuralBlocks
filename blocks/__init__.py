# import sys

# if __package__ is not None:
#     sys.path.insert(1,sys.path[0]+'/'+__package__)

# print(sys.path)

# Perform all local module imports
# from convnormpool import ConvNormPool
# from depthwiseconv import DepthwiseSperableConv

from NeuralBlocks.blocks.convnorm import *
from NeuralBlocks.blocks.convnormrelupool import *
from NeuralBlocks.blocks.convnormrelu import *
from NeuralBlocks.blocks.spectralnorm import *
from NeuralBlocks.blocks.weightnorm import *
from NeuralBlocks.blocks.denseblock import *
from NeuralBlocks.blocks.depthwiseconv import *
from NeuralBlocks.blocks.linnormrelu import *
from NeuralBlocks.blocks.meanspectralnorm import *
from NeuralBlocks.blocks.meanweightnorm import *
from NeuralBlocks.blocks.transconvnormrelu import *
