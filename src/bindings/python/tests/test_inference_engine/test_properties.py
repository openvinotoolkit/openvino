# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import os

from openvino.runtime import Core, Type
from openvino.runtime import properties

@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_properties_core():
    core = Core()

    properties_to_set = dict([
                                properties.enable_profiling(True),
                                properties.cache_dir("./"),
                                # properties.auto_batch_timeout(21), # Unreachable: Bad cast from: N8pybind116objectE to: St3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEN2ov3AnyESt4lessIS5_ESaISt4pairIKS5_S7_EEE
                                # properties.num_streams(9),
                                properties.inference_num_threads(9),
                                # properties.compilation_num_threads(7), # [ NOT_FOUND ] Unsupported property COMPILATION_NUM_THREADS by CPU plugin
                                properties.affinity(properties.Affinity.NONE),
                                properties.hint.inference_precision(Type.f32),
                                # properties.hint.model_priority(properties.hint.Priority.HIGH), # E [ NOT_FOUND ] Unsupported property MODEL_PRIORITY by CPU plugin
                                properties.hint.performance_mode(properties.hint.PerformanceMode.LATENCY),
                                properties.hint.num_requests(12),
                                # properties.hint.model(...), # untested
                                # properties.hint.allow_auto_batching(False), # [ NOT_FOUND ] Unsupported property ALLOW_AUTO_BATCHING by CPU plugin
                                # properties.device.id("9"), # [ NOT_FOUND ] Unsupported property DEVICE_ID by CPU plugin
                                # properties.log.level(properties.log.Level.INFO), # [ NOT_FOUND ] Unsupported property LOG_LEVEL by CPU plugin
                             ])

    core.set_property(properties_to_set)

    # RW properties
    assert core.get_property("CPU", properties.enable_profiling()) == True
    assert core.get_property("CPU", properties.cache_dir()) == "./"
    assert core.get_property("CPU", properties.inference_num_threads()) == 9
    # assert core.get_property("CPU", properties.compilation_num_threads()) == 7
    assert core.get_property("CPU", properties.affinity()) == properties.Affinity.NONE
    assert core.get_property("CPU", properties.hint.inference_precision()) == Type.f32
    assert core.get_property("CPU", properties.hint.performance_mode()) == properties.hint.PerformanceMode.LATENCY
    assert core.get_property("CPU", properties.hint.num_requests()) == 12
    # assert core.get_property("CPU", properties.hint.model()) == ...
    # assert core.get_property("CPU", properties.hint.allow_auto_batching()) == False
    # assert core.get_property("CPU", properties.device.id()) == "9"
    # assert core.get_property("CPU", properties.log.level()) == properties.log.Level.INFO

    # RO properties
    assert type(core.get_property("CPU", properties.supported_properties())) == dict
    assert type(core.get_property("CPU", properties.available_devices())) == list
    # assert core.get_property("CPU", properties.model_name()) # RuntimeError: CPU plugin: . Unsupported config parameter: NETWORK_NAME
    assert type(core.get_property("CPU", properties.optimal_number_of_infer_requests())) == int
    assert type(core.get_property("CPU", properties.range_for_streams())) == tuple
    # assert core.get_property("CPU", properties.optimal_batch_size()) # RuntimeError: CPU plugin: . Unsupported config parameter: OPTIMAL_BATCH_SIZE
    # assert core.get_property("CPU", properties.max_batch_size()) # RuntimeError: CPU plugin: . Unsupported config parameter: MAX_BATCH_SIZE
    assert type(core.get_property("CPU", properties.range_for_async_infer_requests())) == tuple
    assert type(core.get_property("CPU", properties.device.full_name())) == str
    # assert type(core.get_property("CPU", properties.device.architecture())) == str # RuntimeError: CPU plugin: . Unsupported config parameter: DEVICE_ARCHITECTURE
    # assert type(core.get_property("CPU", properties.device.type())) == properties.device.Type # RuntimeError: CPU plugin: . Unsupported config parameter: DEVICE_TYPE
    # assert type(core.get_property("CPU", properties.device.gops())) == ... # RuntimeError: CPU plugin: . Unsupported config parameter: DEVICE_GOPS
    # assert type(core.get_property("CPU", properties.device.thermal())) == float # RuntimeError: CPU plugin: . Unsupported config parameter: DEVICE_THERMAL
    assert type(core.get_property("CPU", properties.device.capabilities())) == list

    # print(core.get_property("CPU", properties.device.capabilities()))
