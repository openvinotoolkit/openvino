from openvino.inference_engine import IECore
from openvino.frontend import FrontEndManager # pylint: disable=import-error
import numpy as np
import os
#import ngraph as ng
from openvino.inference_engine import IENetwork
import openvino

fem = FrontEndManager()
def ov_frontend_run(path_to_pdpd_model: str, user_shapes: dict):
    from openvino.inference_engine import IECore
    #ie_network = ng.function_to_cnn(func)
    ie = IECore()
    return ie.read_network(path_to_pdpd_model)

def ov_frontend_run_(path_to_pdpd_model: str, user_shapes: dict):
    #from ngraph import function_to_cnn # pylint: disable=import-error
    #from ngraph import PartialShape    # pylint: disable=import-error

    print('fem.availableFrontEnds: ' + str(fem.get_available_front_ends()))
    print('Initializing new FE for framework {}'.format("paddle"))
    fe = fem.load_by_framework("paddle")
    print(fe)
    full_model_path = os.path.abspath(path_to_pdpd_model)
    print("Prepare to convert ", full_model_path)

    input_model = fe.load(full_model_path)
    input_places = input_model.get_inputs()

    #for place in input_places:
    #    place_name = place.get_names()[0]
    #    input_model.set_partial_shape(place, PartialShape(user_shapes["shapes"][place_name]))

    model = fe.convert(input_model)

    ie_network = IENetwork(openvino.pyopenvino.Function.to_capsule(model)) #function_to_cnn(model)
    full_model_path = os.path.abspath(path_to_pdpd_model)
    model_name = full_model_path.split('.')[0:-1]
    model_name = '/home/lc/paddle_models/xxx'
    ie_network.serialize(model_name + ".xml", model_name + ".bin")
    print("IR saved.")
    return ie_network

def run_cnn(ie_network, input:dict):
    from openvino.inference_engine import IECore
    #ie_network = ng.function_to_cnn(func)
    ie = IECore()
    executable_network = ie.load_network(ie_network, 'CPU')
    ei = executable_network.get_exec_graph_info()
    ei.serialize('yy.xml')
    output = executable_network.infer(input)
    return output

#ov_frontend_run("/home/lc/paddle_models/MobileNetV1_quant_post", {})
if 0:
    net = ov_frontend_run("/mnt/data/work/openvino_tmp/openvino/bin/intel64/Debug/test_model_zoo/paddle_test_models/fake_depthwise_conv2d_moving_average_abs_max+channel_wise_abs_max/fake_depthwise_conv2d_moving_average_abs_max+channel_wise_abs_max.pdmodel", {})
    r = run_cnn(net, {'x': np.random.random(size=(1, 8, 4, 4))})
if 1:
    net = ov_frontend_run("/home/lc/paddle_models/MobileNetV2_quant_post/", {})
    net.reshape({'read_file_0.tmp_0': [1, 3, 224, 224]})
    r = run_cnn(net, {'read_file_0.tmp_0': np.random.random(size=(1, 3, 224, 224))})
    print(r)

print('done')