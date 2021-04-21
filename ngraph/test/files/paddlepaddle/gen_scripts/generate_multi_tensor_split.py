import paddle
from paddle import fluid
import numpy as np

# it's better to use PYTHON_PATH
# import sys
# sys.path.append('/home/itikhonov/OpenVINO/openvino/bin/intel64/Debug/lib/python_api/python3.6/')
# from openvino.inference_engine import IECore


def create_multi_output_model():
    paddle.enable_static()

    # PDPD model creation and inference
    num_splits = 20
    inp_blob_1 = np.random.randn(2, num_splits, 4, 4).astype(np.float32)

    x = fluid.data(name='x', shape=[2, num_splits, 4, 4], dtype='float32')
    test_layer = fluid.layers.split(x, num_or_sections=num_splits, dim=1)

    var = []
    for i in range(num_splits//2):
        add = fluid.layers.elementwise_add(test_layer[2*i], test_layer[2*i+1])
        var.append(add)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    inp_dict = {'x': inp_blob_1}
    res_pdpd = exe.run(fluid.default_main_program(), fetch_list=var, feed=inp_dict)

    fluid.io.save_inference_model("../models/multi_tensor_split",
                                  list(inp_dict.keys()), var, exe,
                                  model_filename="multi_tensor_split.pdmodel",
                                  params_filename="multi_tensor_split.pdiparams")

    # IE inference
    # ie = IECore()
    # path_to_ie_model = "../models/multi_tensor_split/multi_tensor_split"
    # net = ie.read_network(model=path_to_ie_model + ".xml", weights=path_to_ie_model + ".bin")
    # exec_net = ie.load_network(net, "CPU")
    # res = exec_net.infer({'x': inp_blob_1})
    #
    # # compare results: IE vs PDPD
    # idx = 0
    # for key in res:
    #     comp = np.all(np.isclose(res_pdpd[idx], res[key], rtol=1e-05, atol=1e-08, equal_nan=False))
    #     assert comp, "PDPD and IE results are different"
    #     idx = idx + 1


create_multi_output_model()

