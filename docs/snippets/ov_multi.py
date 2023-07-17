import sys
from openvino.runtime import Core
model_path = "/openvino_CI_CD/result/install_pkg/tests/test_model_zoo/core/models/ir/add_abc.xml"
path_to_model = "/openvino_CI_CD/result/install_pkg/tests/test_model_zoo/core/models/ir/add_abc.xml"
def MULTI_0():
#! [MULTI_0]
    core = Core()

    # Read a network in IR, PaddlePaddle, or ONNX format:
    model = core.read_model(model_path)
    
    # Option 1
    # Pre-configure MULTI globally with explicitly defined devices,
    # and compile the model on MULTI using the newly specified default device list.
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"GPU,CPU"})
    compiled_model = core.compile_model(model=model, device_name="MULTI")

    # Option 2
    # Specify the devices to be used by MULTI explicitly at compilation.
    # The following lines are equivalent:
    compiled_model = core.compile_model(model=model, device_name="MULTI:GPU,CPU")
    compiled_model = core.compile_model(model=model, device_name="MULTI", config={"MULTI_DEVICE_PRIORITIES": "GPU,CPU"}) 

#! [MULTI_0]

def MULTI_1():
#! [MULTI_1]
    core = Core()
    model = core.read_model(model_path)
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"CPU,GPU"})
    # Once the priority list is set, you can alter it on the fly:
    # reverse the order of priorities
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"GPU,CPU"})
    
    # exclude some devices (in this case, CPU)
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"GPU"})
    
    # bring back the excluded devices
    core.set_property(device_name="MULTI", properties={"MULTI_DEVICE_PRIORITIES":"GPU,CPU"})

    # You cannot add new devices on the fly!
    # Attempting to do so will trigger the following exception:
    # [ ERROR ] [NOT_FOUND] You can only change device
    # priorities but not add new devices with the model's
    # ov::device::priorities. CPU device was not in the original device list!

#! [MULTI_1]


# the following two pieces of code appear not to be used anywhere
# they should be considered for removal

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
    dev_match_str = "GPU"
    core = Core()
    model = core.read_model(model_path)
    for d in core.available_devices:
        if dev_match_str in d:
            match_list.append(d)
    all_devices += ",".join(match_list)
    compiled_model = core.compile_model(model=model, device_name=all_devices)
#! [available_devices_2]








def MULTI_4():
#! [MULTI_4]
    core = Core()
    cpu_config = {}
    gpu_config = {}

    # Read a network in IR, PaddlePaddle, or ONNX format:
    model = core.read_model(model_path)

    # When compiling the model on MULTI, configure CPU and GPU 
    # (devices, priorities, and device configurations; gpu_config and cpu_config will load during compile_model() ):
    compiled_model = core.compile_model(model=model, device_name="MULTI:GPU,CPU", config={"CPU":"NUM_STREAMS 4", "GPU":"NUM_STREAMS 8"})

    # Optionally, query the optimal number of requests:
    nireq = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
#! [MULTI_4]

def main():
    MULTI_0()
    MULTI_1()
    available_devices_1()
    available_devices_2()
    MULTI_4()

if __name__ == '__main__':
    sys.exit(main())
