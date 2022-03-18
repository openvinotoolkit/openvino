import sys
from openvino.runtime import Core
model_path = "/openvino_CI_CD/result/install_pkg/tests/test_model_zoo/core/models/ir/add_abc.xml"
path_to_model = "/openvino_CI_CD/result/install_pkg/tests/test_model_zoo/core/models/ir/add_abc.xml"
def Option_1():
#! [Option_1]
    core = Core()

    # Read a network in IR or ONNX format
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model=model, device_name="MULTI:CPU,GPU")
#! [Option_1]

def Option_2():
#! [Option_2]
    core = Core()

    # Read a network in IR or ONNX format
    model = core.read_model(model_path)
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"HDDL,GPU"})
    # Change priorities
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"GPU,HDDL"})
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"GPU"})
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"HDDL,GPU"})
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"CPU,HDDL,GPU"})
#! [Option_2]

def available_devices_1():
#! [available_devices_1]
    all_devices = "MULTI:"
    core = Core()
    model = core.read_model(model_path)
    all_devices += ",".join(core.available_devices)
    compiled_model = core.compile_model(model=model, device_name=all_devices)
#! [available_devices_1]

def available_devices_2():
#! [available_devices_2]
    match_list = []
    all_devices = "MULTI:"
    dev_match_str = "MYRIAD"
    core = Core()
    model = core.read_model(model_path)
    for d in core.available_devices:
        if dev_match_str in d:
            match_list.append(d)
    all_devices += ",".join(match_list)
    compiled_model = core.compile_model(model=model, device_name=all_devices)
#! [available_devices_2]

def set_property():
#! [set_property]
    core = Core()
    cpu_config = {}
    gpu_config = {}
    model = core.read_model(model_path)
    core.set_property(device_name="CPU", properties=cpu_config)
    core.set_property(device_name="GPU", properties=gpu_config)
    compiled_model = core.compile_model(model=model, device_name="MULTI:GPU,CPU")
    # Query the optimal number of requests
    nireq = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
#! [set_property]

def main():
    Option_1()
    Option_2()
    available_devices_1()
    available_devices_2()
    set_property()

if __name__ == '__main__':
    sys.exit(main())
