import paddle
from paddle import fluid
import numpy as np
import os
import sys

paddle.enable_static()

inp_blob = np.random.randn(1, 3, 4, 4).astype(np.float32)

x = fluid.data(name='xxx', shape=[1, 3, 4, 4], dtype='float32')

relu = fluid.layers.relu(x)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
inp_dict = {'xxx': inp_blob}
var = [relu]
res_pdpd = exe.run(fluid.default_main_program(), fetch_list=var, feed=inp_dict)

fluid.io.save_inference_model(os.path.join(sys.argv[1], "relu"), list(inp_dict.keys()), var, exe,
                              model_filename="relu.pdmodel", params_filename="relu.pdiparams")
