import nnvm
import tvm
import mxnet as mx
import numpy as np
import time
import argparse
import json

from tvm import relay
from tvm.contrib import graph_runtime
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model
from gluoncv import model_zoo

import sys

parser = argparse.ArgumentParser(description='Search convolution workload.')
parser.add_argument('--model', type=str, required=True,
                    help="Pretrained model from gluon model zoo.")


run_times = 10

def end2end_benchmark(model, target, batch_size):
    print("Testing %s" % (model))
    num_classes = 1000

    if "ssd" in model:
        image_shape = (3, 512, 512)
        data_shape = (batch_size,) + image_shape
        block = model_zoo.get_model(model, pretrained=True)
        net, params = relay.frontend.from_mxnet(block, shape={'data': data_shape})
    else:
        image_shape = (3, 299, 299) if "inception" in model else (3, 224, 224)
        data_shape = (batch_size,) + image_shape
        block = get_model(model, pretrained=True)
        net, params = relay.frontend.from_mxnet(block, shape={'data': data_shape})
    out_shape = (batch_size, num_classes)

    #tvm.autotvm.task.DispatchContext.current = tvm.autotvm.apply_graph_best("%s_opt.log" % model)
    ctx = tvm.cpu()
    opt_level = 1
    with relay.build_config(opt_level=opt_level):
            graph, lib, params = relay.build_module.build(
                net, target=target, params=params)

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    mx_data = mx.nd.array(data_array)
    module.set_input('data', input_data)

    s = time.time()
    for _ in range(run_times):
        module.run()
    tvm_time = time.time() - s
    print("TVM %s inference time for batch size of %d: %f" % (model, batch_size, tvm_time * 1000/run_times))
    return tvm_time/run_times * 1000


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"
    tvm_resnet = 0
    tvm_mobilenet = 0
    tm= end2end_benchmark(model, target, batch_size)
    tvm_mobilenet += tm


