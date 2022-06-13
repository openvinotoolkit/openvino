# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import os

from openvino.runtime import Core, Type, OVAny
from openvino.runtime import properties


def test_property_rw():
    assert properties.device.priorities() == "MULTI_DEVICE_PRIORITIES"
    assert properties.device.priorities("CPU,GPU") == ("MULTI_DEVICE_PRIORITIES", OVAny("CPU,GPU,"))
    assert properties.device.priorities("CPU", "GPU") == ("MULTI_DEVICE_PRIORITIES", OVAny("CPU,GPU,"))

    with pytest.raises(TypeError) as e:
        value = 6
        properties.device.priorities("CPU", value)
    assert f"Incorrect passed value: {value} , expected string values." in str(e.value)


def test_property_ro():
    assert properties.available_devices() == "AVAILABLE_DEVICES"

    with pytest.raises(TypeError) as e:
        properties.available_devices("something")
    assert "available_devices(): incompatible function arguments." in str(e.value)


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_single_property_setting():
    core = Core()
    core.set_property("CPU", properties.streams.num(properties.streams.Num.AUTO))

    assert properties.streams.Num.AUTO.to_integer() == -1
    assert type(core.get_property("CPU", properties.streams.num())) == int


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
@pytest.mark.parametrize("properties_to_set", [
    # Dict from list of tuples
    dict([  # noqa: C406
        properties.enable_profiling(True),
        properties.cache_dir("./"),
        properties.inference_num_threads(9),
        properties.affinity(properties.Affinity.NONE),
        properties.hint.inference_precision(Type.f32),
        properties.hint.performance_mode(properties.hint.PerformanceMode.LATENCY),
        properties.hint.num_requests(12),
        properties.streams.num(5),
    ]),
    # Pure dict
    {
        properties.enable_profiling(): True,
        properties.cache_dir(): "./",
        properties.inference_num_threads(): 9,
        properties.affinity(): properties.Affinity.NONE,
        properties.hint.inference_precision(): Type.f32,
        properties.hint.performance_mode(): properties.hint.PerformanceMode.LATENCY,
        properties.hint.num_requests(): 12,
        properties.streams.num(): 5,
    },
    # Mixed dict
    {
        properties.enable_profiling(): True,
        "CACHE_DIR": "./",
        properties.inference_num_threads(): 9,
        properties.affinity(): "NONE",
        "INFERENCE_PRECISION_HINT": Type.f32,
        properties.hint.performance_mode(): properties.hint.PerformanceMode.LATENCY,
        properties.hint.num_requests(): 12,
        "NUM_STREAMS": properties.streams.Num(5),
    },
])
def test_properties_core(properties_to_set):
    core = Core()
    core.set_property(properties_to_set)

    # RW properties
    assert core.get_property("CPU", properties.enable_profiling()) is True
    assert core.get_property("CPU", properties.cache_dir()) == "./"
    assert core.get_property("CPU", properties.inference_num_threads()) == 9
    assert core.get_property("CPU", properties.affinity()) == properties.Affinity.NONE
    assert core.get_property("CPU", properties.hint.inference_precision()) == Type.f32
    assert core.get_property("CPU", properties.hint.performance_mode()) == properties.hint.PerformanceMode.LATENCY
    assert core.get_property("CPU", properties.hint.num_requests()) == 12
    assert core.get_property("CPU", properties.streams.num()) == 5

    # RO properties
    assert type(core.get_property("CPU", properties.supported_properties())) == dict
    assert type(core.get_property("CPU", properties.available_devices())) == list
    assert type(core.get_property("CPU", properties.optimal_number_of_infer_requests())) == int
    assert type(core.get_property("CPU", properties.range_for_streams())) == tuple
    assert type(core.get_property("CPU", properties.range_for_async_infer_requests())) == tuple
    assert type(core.get_property("CPU", properties.device.full_name())) == str
    assert type(core.get_property("CPU", properties.device.capabilities())) == list
