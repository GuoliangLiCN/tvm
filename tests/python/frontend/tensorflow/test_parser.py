# 1) nnvm version
import tvm
import nnvm
from tvm.relay.frontend.tensorflow_parser import TFParser
import tensorflow as tf
import numpy as np
#from nnvm.frontend.util.tensorflow_parser import TFParser
# 2) Relay version, need uncomment the block below and comment the block above once relay is adopted
from tvm import relay

def main():
    # model_dir could be the model path to your saved model or frozen single pb
    model_dir = "/Users/yongwu/PycharmProjects/TrialError/placeholder.pb"
    parser = TFParser(model_dir)
    graph_def = parser.parse()
    sym, params = nnvm.frontend.from_tensorflow(graph_def)
    #2) relay, need uncomment the block below and comment the block above once relay is adopted
    #sym, params = relay.frontend.from_tensorflow(graph_def)
if __name__ == '__main__':
    main()
