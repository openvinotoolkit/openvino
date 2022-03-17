import sys
from openvino.runtime import Core
from openvino.inference_engine import IECore
model_path = "/openvino_CI_CD/result/install_pkg/tests/test_model_zoo/core/models/ir/add_abc.xml"
path_to_model = "/openvino_CI_CD/result/install_pkg/tests/test_model_zoo/core/models/ir/add_abc.xml"

def part0():
#! [part0]
    core = Core()

    # Read a network in IR, PaddlePaddle, or ONNX format:
    model = core.read_model(model_path)

    # Load a network to AUTO using the default list of device candidates.
    # The following lines are equivalent:
    compiled_model = core.compile_model(model=model)
    compiled_model = core.compile_model(model=model, device_name="AUTO")
    compiled_model = core.compile_model(model=model, device_name="AUTO", config={})

    # You can also specify the devices to be used by AUTO in its selection process.
    # The following lines are equivalent:
    compiled_model = core.compile_model(model=model, device_name="AUTO:GPU,CPU")
    compiled_model = core.compile_model(model=model, device_name="AUTO", config={"MULTI_DEVICE_PRIORITIES": "GPU,CPU"})

    # the AUTO plugin is pre-configured (globally) with the explicit option:
    core.set_property(device_name="AUTO", properties={"MULTI_DEVICE_PRIORITIES":"GPU,CPU"})
#! [part0]

def part1():
#! [part1]
    ### API before 2022.1 ###
    ie = IECore()

    # Read a network in IR, PaddlePaddle, or ONNX format:
    net = ie.read_network(model=path_to_model)

    # Load a network to AUTO using the default list of device candidates.
    # The following lines are equivalent:
    exec_net = ie.load_network(network=net)
    exec_net = ie.load_network(network=net, device_name="AUTO")
    exec_net = ie.load_network(network=net, device_name="AUTO", config={})

    # You can also specify the devices to be used by AUTO in its selection process.
    # The following lines are equivalent:
    exec_net = ie.load_network(network=net, device_name="AUTO:GPU,CPU")
    exec_net = ie.load_network(network=net, device_name="AUTO", config={"MULTI_DEVICE_PRIORITIES": "GPU,CPU"})

    # the AUTO plugin is pre-configured (globally) with the explicit option:
    ie.set_config(config={"MULTI_DEVICE_PRIORITIES":"GPU,CPU"}, device_name="AUTO");
#! [part1]

def part3():
#! [part3]
    core = Core()
    # Read a network in IR, PaddlePaddle, or ONNX format:
    model = core.read_model(model_path)
    # Load a network to AUTO with Performance Hints enabled:
    # To use the “throughput” mode:
    compiled_model = core.compile_model(model=model, device_name="AUTO:GPU,CPU", config={"PERFORMANCE_HINT":"THROUGHPUT"})
    # or the “latency” mode:
    compiled_model = core.compile_model(model=model, device_name="AUTO:GPU,CPU", config={"PERFORMANCE_HINT":"LATENCY"})
#! [part3]

def part4():
#! [part4]
    core = Core()
    model = core.read_model(model_path)
    compiled_model0 = core.compile_model(model=model, device_name="AUTO:GPU,MYRIAD,CPU", config={"MODEL_PRIORITY":"HIGH"})
    compiled_model1 = core.compile_model(model=model, device_name="AUTO:GPU,MYRIAD,CPU", config={"MODEL_PRIORITY":"MEDIUM"})
    compiled_model2 = core.compile_model(model=model, device_name="AUTO:GPU,MYRIAD,CPU", config={"MODEL_PRIORITY":"LOW"})
    compiled_model0 = core.compile_model(model=model, device_name="AUTO:GPU,MYRIAD,CPU", config={"MODEL_PRIORITY":"HIGH"})
    compiled_model1 = core.compile_model(model=model, device_name="AUTO:GPU,MYRIAD,CPU", config={"MODEL_PRIORITY":"MEDIUM"})
    compiled_model2 = core.compile_model(model=model, device_name="AUTO:GPU,MYRIAD,CPU", config={"MODEL_PRIORITY":"LOW"})
#! [part4]

def part5():
#! [part5]
    core = Core()
    model = core.read_model(model_path)
    core.set_property(device_name="CPU", properties={})
    core.set_property(device_name="MYRIAD", properties={})
    compiled_model = core.compile_model(model=model)
    compiled_model = core.compile_model(model=model, device_name="AUTO")
#! [part5]

def main():
    part0()
    part1()
    part3()
    part4()
    part5()

if __name__ == '__main__':
    sys.exit(main())
