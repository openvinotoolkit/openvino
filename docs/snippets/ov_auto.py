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

    #  compile a model on AUTO using the default list of device candidates.
    #  The following lines are equivalent:
    compiled_model = core.compile_model(model=model)
    compiled_model = core.compile_model(model=model, device_name="AUTO")

    # Optional
    # You can also specify the devices to be used by AUTO.
    # The following lines are equivalent:
    compiled_model = core.compile_model(model=model, device_name="AUTO:GPU,CPU")
    compiled_model = core.compile_model(model=model, device_name="AUTO", config={"MULTI_DEVICE_PRIORITIES": "GPU,CPU"})

    # Optional
    # the AUTO plugin is pre-configured (globally) with the explicit option:
    core.set_property(device_name="AUTO", properties={"MULTI_DEVICE_PRIORITIES":"GPU,CPU"})
#! [part0]

def part1():
#! [part1]
    ### IE API ###
    ie = IECore()

    # Read a network in IR, PaddlePaddle, or ONNX format:
    net = ie.read_network(model=path_to_model)

    # Load a network to AUTO using the default list of device candidates.
    # The following lines are equivalent:
    exec_net = ie.load_network(network=net)
    exec_net = ie.load_network(network=net, device_name="AUTO")
    exec_net = ie.load_network(network=net, device_name="AUTO", config={})

    # Optional
    # You can also specify the devices to be used by AUTO in its selection process.
    # The following lines are equivalent:
    exec_net = ie.load_network(network=net, device_name="AUTO:GPU,CPU")
    exec_net = ie.load_network(network=net, device_name="AUTO", config={"MULTI_DEVICE_PRIORITIES": "GPU,CPU"})

    # Optional
    # the AUTO plugin is pre-configured (globally) with the explicit option:
    ie.set_config(config={"MULTI_DEVICE_PRIORITIES":"GPU,CPU"}, device_name="AUTO");
#! [part1]

def part3():
#! [part3]
    core = Core()
    # Read a network in IR, PaddlePaddle, or ONNX format:
    model = core.read_model(model_path)
    # Compile a model on AUTO with Performance Hints enabled:
    # To use the “THROUGHPUT” mode:
    compiled_model = core.compile_model(model=model, device_name="AUTO", config={"PERFORMANCE_HINT":"THROUGHPUT"})
    # To use the “LATENCY” mode:
    compiled_model = core.compile_model(model=model, device_name="AUTO", config={"PERFORMANCE_HINT":"LATENCY"})
    # To use the “CUMULATIVE_THROUGHPUT” mode:
    compiled_model = core.compile_model(model=model, device_name="AUTO", config={"PERFORMANCE_HINT":"CUMULATIVE_THROUGHPUT"})
#! [part3]

def part4():
#! [part4]
    core = Core()
    model = core.read_model(model_path)

    # Example 1
    compiled_model0 = core.compile_model(model=model, device_name="AUTO", config={"MODEL_PRIORITY":"HIGH"})
    compiled_model1 = core.compile_model(model=model, device_name="AUTO", config={"MODEL_PRIORITY":"MEDIUM"})
    compiled_model2 = core.compile_model(model=model, device_name="AUTO", config={"MODEL_PRIORITY":"LOW"})
    # Assume that all the devices (CPU and GPUs) can support all the networks.
    # Result: compiled_model0 will use GPU.1, compiled_model1 will use GPU.0, compiled_model2 will use CPU.

    # Example 2
    compiled_model3 = core.compile_model(model=model, device_name="AUTO", config={"MODEL_PRIORITY":"HIGH"})
    compiled_model4 = core.compile_model(model=model, device_name="AUTO", config={"MODEL_PRIORITY":"MEDIUM"})
    compiled_model5 = core.compile_model(model=model, device_name="AUTO", config={"MODEL_PRIORITY":"LOW"})
    # Assume that all the devices (CPU ang GPUs) can support all the networks.
    # Result: compiled_model3 will use GPU.1, compiled_model4 will use GPU.1, compiled_model5 will use GPU.0.
#! [part4]

def part5():
#! [part5]
    core = Core()
    model = core.read_model(model_path)
    # gpu_config and cpu_config will load during compile_model()
    compiled_model = core.compile_model(model=model)
    compiled_model = core.compile_model(model=model, device_name="AUTO")
#! [part5]

def part6():
#! [part6]
    core = Core()
    # read a network in IR, PaddlePaddle, or ONNX format
    model = core.read_model(model_path)
    # compile a model on AUTO and set log level to debug
    compiled_model = core.compile_model(model=model, device_name="AUTO", config={"LOG_LEVEL":"LOG_DEBUG"});
    # set log level with set_property and compile model
    core.set_property(device_name="AUTO", properties={"LOG_LEVEL":"LOG_DEBUG"});
    compiled_model = core.compile_model(model=model, device_name="AUTO");
#! [part6]

def part7():
#! [part7]
    core = Core()
    # read a network in IR, PaddlePaddle, or ONNX format
    model = core.read_model(model_path)
    # compile a model on AUTO and set log level to debug
    compiled_model = core.compile_model(model=model, device_name="AUTO")
    # query the runtime target devices on which the inferences are being executed
    execution_devices = compiled_model.get_property("EXECUTION_DEVICES")
#! [part7]

def main():
    part0()
    part1()
    part3()
    part4()
    part5()
    part6()
    part7()

if __name__ == '__main__':
    sys.exit(main())
