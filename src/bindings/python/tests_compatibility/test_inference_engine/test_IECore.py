# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import pytest
import sys
from pathlib import Path
from threading import Event, Thread
from time import sleep, time
from queue import Queue

from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork
from tests_compatibility.conftest import model_path, plugins_path, model_onnx_path
import ngraph as ng


test_net_xml, test_net_bin = model_path()
test_net_onnx = model_onnx_path()
plugins_xml, plugins_win_xml, plugins_osx_xml = plugins_path()


def test_init_ie_core_no_cfg():
    ie = IECore()
    assert isinstance(ie, IECore)


def test_init_ie_core_with_cfg():
    ie = IECore(plugins_xml)
    assert isinstance(ie, IECore)


def test_get_version(device):
    ie = IECore()
    version = ie.get_versions(device)
    assert isinstance(version, dict), "Returned version must be a dictionary"
    assert device in version, "{} plugin version wasn't found in versions"
    assert hasattr(version[device], "major"), "Returned version has no field 'major'"
    assert hasattr(version[device], "minor"), "Returned version has no field 'minor'"
    assert hasattr(version[device], "description"), "Returned version has no field 'description'"
    assert hasattr(version[device], "build_number"), "Returned version has no field 'build_number'"


def test_load_network(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device)
    assert isinstance(exec_net, ExecutableNetwork)

def test_load_network_without_device():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net)
    assert isinstance(exec_net, ExecutableNetwork)

def test_load_network_from_file(device):
    ie = IECore()
    exec_net = ie.load_network(test_net_xml, device)
    assert isinstance(exec_net, ExecutableNetwork)

def test_load_network_from_file_without_device():
    ie = IECore()
    exec_net = ie.load_network(test_net_xml)
    assert isinstance(exec_net, ExecutableNetwork)

@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_load_network_wrong_device():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(RuntimeError) as e:
        ie.load_network(net, "BLA")
    assert 'Device with "BLA" name is not registered in the OpenVINO Runtime' in str(e.value)


def test_query_network(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    query_res = ie.query_network(net, device)
    func_net = ng.function_from_cnn(net)
    ops_net = func_net.get_ordered_ops()
    ops_net_names = [op.friendly_name for op in ops_net]
    assert [key for key in query_res.keys() if key not in ops_net_names] == [], \
        "Not all network layers present in query_network results"
    assert next(iter(set(query_res.values()))) == device, "Wrong device for some layers"


@pytest.mark.dynamic_library
def test_register_plugin():
    device = "TEST_DEVICE"
    lib_name = "test_plugin"
    full_lib_name = lib_name + ".dll" if sys.platform == "win32" else "lib" + lib_name + ".so"

    ie = IECore()
    ie.register_plugin(lib_name, device)
    with pytest.raises(RuntimeError) as e:
        ie.get_versions(device)
    assert f"Cannot load library '{full_lib_name}'" in str(e.value)

@pytest.mark.dynamic_library
def test_register_plugins():
    device = "TEST_DEVICE"
    lib_name = "test_plugin"
    full_lib_name = lib_name + ".dll" if sys.platform == "win32" else "lib" + lib_name + ".so"
    plugins_xml_path = os.path.join(os.getcwd(), "plugin_path.xml")

    plugin_xml = f"""<ie>
    <plugins>
        <plugin location="{full_lib_name}" name="{device}">
        </plugin>
    </plugins>
    </ie>"""

    with open(plugins_xml_path, "w") as f:
        f.write(plugin_xml)
    
    ie = IECore()
    ie.register_plugins(plugins_xml_path)
    os.remove(plugins_xml_path)

    with pytest.raises(RuntimeError) as e:
        ie.get_versions(device)
    assert f"Cannot load library '{full_lib_name}'" in str(e.value)


def test_unload_plugin(device):
    ie = IECore()
    # Trigger plugin loading
    ie.get_versions(device)
    # Unload plugin
    ie.unregister_plugin(device)


def test_available_devices(device):
    ie = IECore()
    devices = ie.available_devices
    assert device in devices, f"Current device '{device}' is not listed in available devices '{', '.join(devices)}'"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_list_of_str():
    ie = IECore()
    param = ie.get_metric("CPU", "OPTIMIZATION_CAPABILITIES")
    assert isinstance(param, list), "Parameter value for 'OPTIMIZATION_CAPABILITIES' " \
                                    f"metric must be a list but {type(param)} is returned"
    assert all(isinstance(v, str) for v in param), "Not all of the parameter values for 'OPTIMIZATION_CAPABILITIES' " \
                                                   "metric are strings!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_two_ints():
    ie = IECore()
    if ie.get_metric("CPU", "FULL_DEVICE_NAME") == "arm_compute::NEON":
        pytest.skip("Can't run on ARM plugin due-to unsupported device metric")
    param = ie.get_metric("CPU", "RANGE_FOR_STREAMS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_STREAMS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for 'RANGE_FOR_STREAMS' " \
                                                   "metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_three_ints():
    ie = IECore()
    if ie.get_metric("CPU", "FULL_DEVICE_NAME") == "arm_compute::NEON":
        pytest.skip("Can't run on ARM plugin due-to unsupported device metric")
    param = ie.get_metric("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_ASYNC_INFER_REQUESTS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for " \
                                                   "'RANGE_FOR_ASYNC_INFER_REQUESTS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_str():
    ie = IECore()
    param = ie.get_metric("CPU", "FULL_DEVICE_NAME")
    assert isinstance(param, str), "Parameter value for 'FULL_DEVICE_NAME' " \
                                   f"metric must be string but {type(param)} is returned"


def test_read_network_from_xml():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net, IENetwork)

    net = ie.read_network(model=test_net_xml)
    assert isinstance(net, IENetwork)


def test_read_network_as_path():
    ie = IECore()

    net = ie.read_network(model=Path(test_net_xml), weights=test_net_bin)
    assert isinstance(net, IENetwork)

    net = ie.read_network(model=test_net_xml, weights=Path(test_net_bin))
    assert isinstance(net, IENetwork)

    net = ie.read_network(model=Path(test_net_xml))
    assert isinstance(net, IENetwork)


def test_read_network_from_onnx():
    ie = IECore()
    net = ie.read_network(model=test_net_onnx)
    assert isinstance(net, IENetwork)


def test_read_network_from_onnx_as_path():
    ie = IECore()
    net = ie.read_network(model=Path(test_net_onnx))
    assert isinstance(net, IENetwork)


def test_incorrect_xml():
    ie = IECore()
    with pytest.raises(Exception) as e:
        ie.read_network(model="./model.xml", weights=Path(test_net_bin))
    assert "Path to the model ./model.xml doesn't exist or it's a directory" in str(e.value)


def test_incorrect_bin():
    ie = IECore()
    with pytest.raises(Exception) as e:
        ie.read_network(model=test_net_xml, weights="./model.bin")
    assert "Path to the weights ./model.bin doesn't exist or it's a directory" in str(e.value)


def test_read_net_from_buffer():
    ie = IECore()
    with open(test_net_bin, 'rb') as f:
        bin = f.read()
    with open(model_path()[0], 'rb') as f:
        xml = f.read()
    net = ie.read_network(model=xml, weights=bin, init_from_buffer=True)
    assert isinstance(net, IENetwork)


def test_net_from_buffer_valid():
    ie = IECore()
    with open(test_net_bin, 'rb') as f:
        bin = f.read()
    with open(model_path()[0], 'rb') as f:
        xml = f.read()
    net = ie.read_network(model=xml, weights=bin, init_from_buffer=True)
    ref_net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.name == ref_net.name
    assert net.batch_size == ref_net.batch_size
    ii_net = net.input_info
    ii_net2 = ref_net.input_info
    o_net = net.outputs
    o_net2 = ref_net.outputs
    assert ii_net.keys() == ii_net2.keys()
    assert o_net.keys() == o_net2.keys()


@pytest.mark.skipif(os.environ.get("TEST_DEVICE","CPU") != "GPU", reason=f"Device dependent test")
def test_load_network_release_gil(device):
    running = True
    message_queue = Queue()
    def detect_long_gil_holds():
        sleep_time = 0.01
        latency_alert_threshold = 0.1
        # Send a message to indicate the thread is running and ready to detect GIL locks
        message_queue.put("ready to detect")
        while running:
            start_sleep = time()
            sleep(sleep_time)
            elapsed = time() - start_sleep
            if elapsed > latency_alert_threshold:
                # Send a message to the testing thread that a long GIL lock occurred
                message_queue.put(latency_alert_threshold)
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    # Wait for the GIL lock detector to be up and running
    gil_hold_detection_thread = Thread(daemon=True, target=detect_long_gil_holds)
    gil_hold_detection_thread.start()
    # Wait to make sure the thread is started and checking for GIL holds
    sleep(0.1)
    assert message_queue.get(timeout=5) == "ready to detect"
    # Run the function that should unlock the GIL
    exec_net = ie.load_network(net, device)
    # Ensure resources are closed
    running = False
    gil_hold_detection_thread.join(timeout=5)
    # Assert there were never any long gil locks
    assert message_queue.qsize() == 0, \
        f"More than 0 GIL locks occured! Latency: {message_queue.get()})"


def test_nogil_safe(device):
    libc_name, libc_version = platform.libc_ver()
    if libc_name == 'glibc':
        version = tuple(int(x) for x in libc_version.split('.'))
        if version < (2, 34):
            pytest.skip("There is an issue in glibc for an older version.")

    call_thread_func = Event()
    switch_interval = sys.getswitchinterval()
    core = IECore()
    net = core.read_network(model=test_net_xml, weights=test_net_bin)

    def thread_target(thread_func, thread_args):
        call_thread_func.wait()
        call_thread_func.clear()
        thread_func(*thread_args)

    def main_thread_target(gil_release_func, args):
        call_thread_func.set()
        gil_release_func(*args)

    def test_run_parallel(gil_release_func, args, thread_func, thread_args):
        thread = Thread(target=thread_target, args=[thread_func, thread_args])
        sys.setswitchinterval(1000)
        thread.start()
        main_thread_target(gil_release_func, args)
        thread.join()
        sys.setswitchinterval(switch_interval)

    main_targets = [{
                     core.read_network: [test_net_xml, test_net_bin],
                     core.load_network: [net, device],
                    },
                    {
                     core.load_network: [net, device],
                    }]

    thread_targets = [{
                       core.get_versions: [device,],
                       core.read_network: [test_net_xml, test_net_bin],
                       core.load_network: [net, device],
                       core.query_network: [net, device],
                       getattr: [core, "available_devices"],
                      },
                      {
                       getattr: [net, "name"],
                       getattr: [net, "input_info"],
                       getattr: [net, "outputs"],
                       getattr: [net, "batch_size"],
                      }]

    for main_target, custom_target in zip(main_targets, thread_targets):
        for nogil_func, args in main_target.items():
            for thread_func, thread_args in custom_target.items():
                test_run_parallel(nogil_func, args, thread_func, thread_args)
