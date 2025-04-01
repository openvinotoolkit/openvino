# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import os

import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.intel_cpu as intel_cpu
import openvino.properties.intel_auto as intel_auto
import openvino.properties.intel_gpu as intel_gpu
import openvino.properties.intel_gpu.hint as intel_gpu_hint
import openvino.properties.intel_npu as intel_npu
import openvino.properties.device as device
import openvino.properties.log as log
import openvino.properties.streams as streams
from openvino import Core, Type, OVAny
from openvino import properties


###
# Base properties API
###
def test_properties_ro_base():
    with pytest.raises(TypeError) as e:
        props.supported_properties("something")
    assert "incompatible function arguments" in str(e.value)


def test_properties_rw_base():
    assert ov.properties.cache_dir == "CACHE_DIR"
    assert props.cache_dir("./test_dir") == ("CACHE_DIR", OVAny("./test_dir"))
    assert properties.cache_dir("./test_dir") == ("CACHE_DIR", OVAny("./test_dir"))

    with pytest.raises(TypeError) as e:
        props.cache_dir(6)
    assert "incompatible function arguments" in str(e.value)


###
# Enum-like values
###
@pytest.mark.parametrize(
    ("ov_enum", "expected_values"),
    [
        (
            props.CacheMode,
            (
                (props.CacheMode.OPTIMIZE_SIZE, "CacheMode.OPTIMIZE_SIZE", 0),
                (props.CacheMode.OPTIMIZE_SPEED, "CacheMode.OPTIMIZE_SPEED", 1),
            ),
        ),
        (
            props.WorkloadType,
            (
                (props.WorkloadType.DEFAULT, "WorkloadType.DEFAULT", 0),
                (props.WorkloadType.EFFICIENT, "WorkloadType.EFFICIENT", 1),
            ),
        ),
        (
            hints.Priority,
            (
                (hints.Priority.LOW, "Priority.LOW", 0),
                (hints.Priority.MEDIUM, "Priority.MEDIUM", 1),
                (hints.Priority.HIGH, "Priority.HIGH", 2),
                (hints.Priority.DEFAULT, "Priority.MEDIUM", 1),
            ),
        ),
        (
            hints.PerformanceMode,
            (
                (hints.PerformanceMode.LATENCY, "PerformanceMode.LATENCY", 1),
                (hints.PerformanceMode.THROUGHPUT, "PerformanceMode.THROUGHPUT", 2),
                (hints.PerformanceMode.CUMULATIVE_THROUGHPUT, "PerformanceMode.CUMULATIVE_THROUGHPUT", 3),
            ),
        ),
        (
            hints.SchedulingCoreType,
            (
                (hints.SchedulingCoreType.ANY_CORE, "SchedulingCoreType.ANY_CORE", 0),
                (hints.SchedulingCoreType.PCORE_ONLY, "SchedulingCoreType.PCORE_ONLY", 1),
                (hints.SchedulingCoreType.ECORE_ONLY, "SchedulingCoreType.ECORE_ONLY", 2),
            ),
        ),
        (
            hints.ModelDistributionPolicy,
            (
                (hints.ModelDistributionPolicy.TENSOR_PARALLEL, "ModelDistributionPolicy.TENSOR_PARALLEL", 0),
            ),
        ),
        (
            hints.ExecutionMode,
            (
                (hints.ExecutionMode.PERFORMANCE, "ExecutionMode.PERFORMANCE", 1),
                (hints.ExecutionMode.ACCURACY, "ExecutionMode.ACCURACY", 2),
            ),
        ),
        (
            device.Type,
            (
                (device.Type.INTEGRATED, "Type.INTEGRATED", 0),
                (device.Type.DISCRETE, "Type.DISCRETE", 1),
            ),
        ),
        (
            log.Level,
            (
                (log.Level.NO, "Level.NO", -1),
                (log.Level.ERR, "Level.ERR", 0),
                (log.Level.WARNING, "Level.WARNING", 1),
                (log.Level.INFO, "Level.INFO", 2),
                (log.Level.DEBUG, "Level.DEBUG", 3),
                (log.Level.TRACE, "Level.TRACE", 4),
            ),
        ),
        (
            intel_auto.SchedulePolicy,
            (
                (intel_auto.SchedulePolicy.ROUND_ROBIN, "SchedulePolicy.ROUND_ROBIN", 0),
                (intel_auto.SchedulePolicy.DEVICE_PRIORITY, "SchedulePolicy.DEVICE_PRIORITY", 1),
                (intel_auto.SchedulePolicy.DEFAULT, "SchedulePolicy.DEVICE_PRIORITY", 1),
            ),
        ),
    ],
)
def test_properties_enums(ov_enum, expected_values):
    assert ov_enum is not None
    enum_entries = iter(ov_enum.__entries.values())

    for property_obj, property_str, property_int in expected_values:
        assert property_obj == next(enum_entries)[0]
        assert str(property_obj) == property_str
        assert int(property_obj) == property_int


@pytest.mark.parametrize(
    ("proxy_enums", "expected_values"),
    [
        (
            (
                intel_gpu_hint.ThrottleLevel.LOW,
                intel_gpu_hint.ThrottleLevel.MEDIUM,
                intel_gpu_hint.ThrottleLevel.HIGH,
                intel_gpu_hint.ThrottleLevel.DEFAULT,
            ),
            (
                ("Priority.LOW", 0),
                ("Priority.MEDIUM", 1),
                ("Priority.HIGH", 2),
                ("Priority.MEDIUM", 1),
            ),
        ),
    ],
)
def test_conflicting_enum(proxy_enums, expected_values):
    assert len(proxy_enums) == len(expected_values)

    for i in range(len(proxy_enums)):
        assert str(proxy_enums[i]) == expected_values[i][0]
        assert int(proxy_enums[i]) == expected_values[i][1]


###
# Read-Only properties
###
@pytest.mark.parametrize(
    ("ov_property_ro", "expected_value"),
    [
        (props.supported_properties, "SUPPORTED_PROPERTIES"),
        (props.available_devices, "AVAILABLE_DEVICES"),
        (props.model_name, "NETWORK_NAME"),
        (props.optimal_number_of_infer_requests, "OPTIMAL_NUMBER_OF_INFER_REQUESTS"),
        (props.range_for_streams, "RANGE_FOR_STREAMS"),
        (props.optimal_batch_size, "OPTIMAL_BATCH_SIZE"),
        (props.max_batch_size, "MAX_BATCH_SIZE"),
        (props.range_for_async_infer_requests, "RANGE_FOR_ASYNC_INFER_REQUESTS"),
        (props.execution_devices, "EXECUTION_DEVICES"),
        (props.loaded_from_cache, "LOADED_FROM_CACHE"),
        (device.full_name, "FULL_DEVICE_NAME"),
        (device.architecture, "DEVICE_ARCHITECTURE"),
        (device.type, "DEVICE_TYPE"),
        (device.gops, "DEVICE_GOPS"),
        (device.thermal, "DEVICE_THERMAL"),
        (device.uuid, "DEVICE_UUID"),
        (device.luid, "DEVICE_LUID"),
        (device.capabilities, "OPTIMIZATION_CAPABILITIES"),
        (intel_gpu.device_total_mem_size, "GPU_DEVICE_TOTAL_MEM_SIZE"),
        (intel_gpu.uarch_version, "GPU_UARCH_VERSION"),
        (intel_gpu.execution_units_count, "GPU_EXECUTION_UNITS_COUNT"),
        (intel_gpu.memory_statistics, "GPU_MEMORY_STATISTICS"),
        (intel_npu.device_alloc_mem_size, "NPU_DEVICE_ALLOC_MEM_SIZE"),
        (intel_npu.device_total_mem_size, "NPU_DEVICE_TOTAL_MEM_SIZE"),
        (intel_npu.driver_version, "NPU_DRIVER_VERSION"),
        (intel_npu.compiler_version, "NPU_COMPILER_VERSION"),
    ],
)
def test_properties_ro(ov_property_ro, expected_value):
    # Test if property is correctly registered
    assert ov_property_ro() == expected_value
    assert ov_property_ro == expected_value


###
# Read-Write properties
###
@pytest.mark.parametrize(
    ("ov_property_rw", "expected_value", "test_values"),
    [
        (
            props.enable_profiling,
            "PERF_COUNT",
            (
                (True, True),
                (False, False),
                (1, True),
                (0, False),
            ),
        ),
        (
            props.cache_dir,
            "CACHE_DIR",
            (("./test_cache", "./test_cache"),),
        ),
        (
            props.cache_mode,
            "CACHE_MODE",
            (
                (props.CacheMode.OPTIMIZE_SIZE, props.CacheMode.OPTIMIZE_SIZE),
                (props.CacheMode.OPTIMIZE_SPEED, props.CacheMode.OPTIMIZE_SPEED),
            ),
        ),
        (
            props.auto_batch_timeout,
            "AUTO_BATCH_TIMEOUT",
            (
                (21, 21),
                (np.uint32(37), 37),
                (21, np.uint32(21)),
                (np.uint32(37), np.uint32(37)),
            ),
        ),
        (
            props.inference_num_threads,
            "INFERENCE_NUM_THREADS",
            (
                (-8, -8),
                (8, 8),
            ),
        ),
        (
            props.compilation_num_threads,
            "COMPILATION_NUM_THREADS",
            ((44, 44),),
        ),
        (props.force_tbb_terminate, "FORCE_TBB_TERMINATE", ((True, True), (False, False))),
        (props.enable_mmap, "ENABLE_MMAP", ((True, True), (False, False))),
        (
            props.weights_path,
            "WEIGHTS_PATH",
            (("./model.bin", "./model.bin"),),
        ),
        (
            props.key_cache_group_size,
            "KEY_CACHE_GROUP_SIZE",
            ((64, 64),),
        ),
        (
            props.value_cache_group_size,
            "VALUE_CACHE_GROUP_SIZE",
            ((64, 64),),
        ),
        (props.key_cache_precision, "KEY_CACHE_PRECISION", ((Type.f32, Type.f32),)),
        (props.value_cache_precision, "VALUE_CACHE_PRECISION", ((Type.f32, Type.f32),)),
        (hints.inference_precision, "INFERENCE_PRECISION_HINT", ((Type.f32, Type.f32),)),
        (
            hints.model_priority,
            "MODEL_PRIORITY",
            ((hints.Priority.LOW, hints.Priority.LOW),),
        ),
        (
            hints.performance_mode,
            "PERFORMANCE_HINT",
            ((hints.PerformanceMode.LATENCY, hints.PerformanceMode.LATENCY),),
        ),
        (
            hints.enable_cpu_pinning,
            "ENABLE_CPU_PINNING",
            (
                (True, True),
                (False, False),
                (1, True),
                (0, False),
            ),
        ),
        (
            hints.scheduling_core_type,
            "SCHEDULING_CORE_TYPE",
            ((hints.SchedulingCoreType.PCORE_ONLY, hints.SchedulingCoreType.PCORE_ONLY),),
        ),
        (
            hints.model_distribution_policy,
            "MODEL_DISTRIBUTION_POLICY",
            (
                ({hints.ModelDistributionPolicy.TENSOR_PARALLEL}, {hints.ModelDistributionPolicy.TENSOR_PARALLEL}),
            ),
        ),
        (
            hints.enable_hyper_threading,
            "ENABLE_HYPER_THREADING",
            (
                (True, True),
                (False, False),
                (1, True),
                (0, False),
            ),
        ),
        (
            hints.execution_mode,
            "EXECUTION_MODE_HINT",
            ((hints.ExecutionMode.PERFORMANCE, hints.ExecutionMode.PERFORMANCE),),
        ),
        (
            hints.num_requests,
            "PERFORMANCE_HINT_NUM_REQUESTS",
            ((8, 8),),
        ),
        (
            hints.allow_auto_batching,
            "ALLOW_AUTO_BATCHING",
            ((True, True),),
        ),
        (
            hints.dynamic_quantization_group_size,
            "DYNAMIC_QUANTIZATION_GROUP_SIZE",
            ((64, 64),),
        ),
        (hints.kv_cache_precision, "KV_CACHE_PRECISION", ((Type.f32, Type.f32),)),
        (
            hints.activations_scale_factor,
            "ACTIVATIONS_SCALE_FACTOR",
            ((0.0, 0.0),),
        ),
        (
            intel_cpu.denormals_optimization,
            "CPU_DENORMALS_OPTIMIZATION",
            ((True, True),),
        ),
        (
            intel_cpu.sparse_weights_decompression_rate,
            "CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE",
            (
                (0.1, np.float32(0.1)),
                (2.0, 2.0),
            ),
        ),
        (
            intel_auto.device_bind_buffer,
            "DEVICE_BIND_BUFFER",
            (
                (True, True),
                (False, False),
                (1, True),
                (0, False),
            ),
        ),
        (
            intel_auto.enable_startup_fallback,
            "ENABLE_STARTUP_FALLBACK",
            (
                (True, True),
                (False, False),
                (1, True),
                (0, False),
            ),
        ),
        (
            intel_auto.enable_runtime_fallback,
            "ENABLE_RUNTIME_FALLBACK",
            (
                (True, True),
                (False, False),
                (1, True),
                (0, False),
            ),
        ),
        (device.id, "DEVICE_ID", (("0", "0"),)),
        (
            log.level,
            "LOG_LEVEL",
            ((log.Level.NO, log.Level.NO),),
        ),
        (
            intel_gpu.enable_loop_unrolling,
            "GPU_ENABLE_LOOP_UNROLLING",
            ((True, True),),
        ),
        (
            intel_gpu.disable_winograd_convolution,
            "GPU_DISABLE_WINOGRAD_CONVOLUTION",
            ((True, True),),
        ),
        (
            intel_gpu_hint.queue_throttle,
            "GPU_QUEUE_THROTTLE",
            ((intel_gpu_hint.ThrottleLevel.LOW, hints.Priority.LOW),),
        ),
        (
            intel_gpu_hint.queue_priority,
            "GPU_QUEUE_PRIORITY",
            ((hints.Priority.LOW, hints.Priority.LOW),),
        ),
        (
            intel_gpu_hint.host_task_priority,
            "GPU_HOST_TASK_PRIORITY",
            ((hints.Priority.LOW, hints.Priority.LOW),),
        ),
        (
            intel_gpu_hint.available_device_mem,
            "AVAILABLE_DEVICE_MEM_SIZE",
            ((128, 128),),
        ),
        (
            intel_npu.compilation_mode_params,
            "NPU_COMPILATION_MODE_PARAMS",
            (("dummy-op-replacement=true", "dummy-op-replacement=true"),),
        ),
        (
            intel_npu.turbo,
            "NPU_TURBO",
            ((True, True),),
        ),
        (
            intel_npu.tiles,
            "NPU_TILES",
            ((128, 128),),
        ),
        (
            intel_npu.max_tiles,
            "NPU_MAX_TILES",
            ((128, 128),),
        ),
        (
            intel_npu.bypass_umd_caching,
            "NPU_BYPASS_UMD_CACHING",
            ((True, True),),
        ),
        (
            intel_npu.defer_weights_load,
            "NPU_DEFER_WEIGHTS_LOAD",
            ((True, True),),
        ),
        (
            intel_npu.compiler_dynamic_quantization,
            "NPU_COMPILER_DYNAMIC_QUANTIZATION",
            ((True, True),),
        ),
    ],
)
def test_properties_rw(ov_property_rw, expected_value, test_values):
    # Test if property is correctly registered
    assert ov_property_rw() == expected_value
    assert ov_property_rw == expected_value

    # Test if property process values correctly
    for values in test_values:
        property_tuple = ov_property_rw(values[0])
        assert property_tuple[0] == expected_value
        assert property_tuple[1].value == values[1]


###
# Special cases
###
def test_compiled_blob_property():
    assert hints.compiled_blob == "COMPILED_BLOB"
    compiled_blob = hints.compiled_blob(ov.Tensor(Type.u8, [2, 5]))

    assert compiled_blob[0] == "COMPILED_BLOB"
    assert compiled_blob[1].value.element_type == Type.u8
    assert compiled_blob[1].value.shape == [2, 5]


def test_properties_device_priorities():
    assert device.priorities == "MULTI_DEVICE_PRIORITIES"
    assert device.priorities("CPU,GPU") == ("MULTI_DEVICE_PRIORITIES", OVAny("CPU,GPU,"))
    assert device.priorities("CPU", "GPU") == ("MULTI_DEVICE_PRIORITIES", OVAny("CPU,GPU,"))

    with pytest.raises(TypeError) as e:
        value = 6
        device.priorities("CPU", value)
    assert f"Incorrect passed value: {value} , expected string values." in str(e.value)


def test_properties_device_properties():
    assert device.properties() == "DEVICE_PROPERTIES"

    def make_dict(*arg):
        return dict(  # noqa: C406
            [*arg])

    def check(value1, value2):
        assert device.properties(value1) == ("DEVICE_PROPERTIES", OVAny(value2))

    check({"CPU": {streams.num: 2}},
          {"CPU": {"NUM_STREAMS": 2}})
    check({"CPU": make_dict(streams.num(2))},
          {"CPU": {"NUM_STREAMS": streams.Num(2)}})
    check({"GPU": make_dict(hints.inference_precision(Type.f32))},
          {"GPU": {"INFERENCE_PRECISION_HINT": Type.f32}})
    check({"CPU": make_dict(streams.num(2), hints.inference_precision(Type.f32))},
          {"CPU": {"INFERENCE_PRECISION_HINT": Type.f32, "NUM_STREAMS": streams.Num(2)}})
    check({"CPU": make_dict(streams.num(2), hints.inference_precision(Type.f32)),
           "GPU": make_dict(streams.num(1), hints.inference_precision(Type.f16))},
          {"CPU": {"INFERENCE_PRECISION_HINT": Type.f32, "NUM_STREAMS": streams.Num(2)},
           "GPU": {"INFERENCE_PRECISION_HINT": Type.f16, "NUM_STREAMS": streams.Num(1)}})


def test_properties_streams():
    # Test extra Num class
    assert streams.Num().to_integer() == -1
    assert streams.Num(2).to_integer() == 2
    assert streams.Num.AUTO.to_integer() == -1
    assert streams.Num.NUMA.to_integer() == -2
    # Test RW property
    property_tuple = streams.num(streams.Num.AUTO)
    assert property_tuple[0] == "NUM_STREAMS"
    assert property_tuple[1].value == -1

    property_tuple = streams.num(42)
    assert property_tuple[0] == "NUM_STREAMS"
    assert property_tuple[1].value == 42


def test_properties_capability():
    assert device.Capability.FP32 == "FP32"
    assert device.Capability.BF16 == "BF16"
    assert device.Capability.FP16 == "FP16"
    assert device.Capability.INT8 == "INT8"
    assert device.Capability.INT16 == "INT16"
    assert device.Capability.BIN == "BIN"
    assert device.Capability.WINOGRAD == "WINOGRAD"
    assert device.Capability.EXPORT_IMPORT == "EXPORT_IMPORT"


def test_properties_memory_type_gpu():
    assert intel_gpu.MemoryType.surface == "GPU_SURFACE"
    assert intel_gpu.MemoryType.buffer == "GPU_BUFFER"


def test_properties_capability_gpu():
    assert intel_gpu.CapabilityGPU.HW_MATMUL == "GPU_HW_MATMUL"


def test_properties_hint_model():
    # Temporary imports
    from tests.utils.helpers import generate_add_model

    model = generate_add_model()

    assert properties.hint.model() == "MODEL_PTR"
    assert hints.model == "MODEL_PTR"

    property_tuple = hints.model(model)
    assert property_tuple[0] == "MODEL_PTR"


def test_single_property_setting(device):
    core = Core()

    core.set_property(device, streams.num(streams.Num.AUTO))

    assert props.streams.Num.AUTO.to_integer() == -1
    assert isinstance(core.get_property(device, streams.num()), int)


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
@pytest.mark.parametrize(
    "properties_to_set",
    [
        # Dict from list of tuples
        dict(  # noqa: C406
            [  # noqa: C406
                props.enable_profiling(True),
                props.cache_dir("./"),
                props.inference_num_threads(9),
                hints.inference_precision(Type.f32),
                hints.performance_mode(hints.PerformanceMode.LATENCY),
                hints.enable_cpu_pinning(True),
                hints.scheduling_core_type(hints.SchedulingCoreType.PCORE_ONLY),
                hints.enable_hyper_threading(True),
                hints.num_requests(12),
                streams.num(5),
            ],
        ),
        # Pure dict
        {
            props.enable_profiling: True,
            props.cache_dir: "./",
            props.inference_num_threads: 9,
            hints.inference_precision: Type.f32,
            hints.performance_mode: hints.PerformanceMode.LATENCY,
            hints.enable_cpu_pinning: True,
            hints.scheduling_core_type: hints.SchedulingCoreType.PCORE_ONLY,
            hints.enable_hyper_threading: True,
            hints.num_requests: 12,
            streams.num: 5,
        },
        # Mixed dict
        {
            props.enable_profiling: True,
            "CACHE_DIR": "./",
            props.inference_num_threads: 9,
            "INFERENCE_PRECISION_HINT": Type.f32,
            hints.performance_mode: hints.PerformanceMode.LATENCY,
            hints.scheduling_core_type: hints.SchedulingCoreType.PCORE_ONLY,
            hints.num_requests: 12,
            "NUM_STREAMS": streams.Num(5),
            "ENABLE_MMAP": False,
        },
    ],
)
def test_core_cpu_properties(properties_to_set):
    core = Core()

    if "Intel" not in core.get_property("CPU", "FULL_DEVICE_NAME"):
        pytest.skip("This test runs only on openvino intel cpu plugin")
    core.set_property(properties_to_set)

    # RW properties
    assert core.get_property("CPU", props.enable_profiling) is True
    assert core.get_property("CPU", props.cache_dir) == "./"
    assert core.get_property("CPU", props.inference_num_threads) == 9
    assert core.get_property("CPU", streams.num) == 5

    # RO properties
    assert isinstance(core.get_property("CPU", props.supported_properties), dict)
    assert isinstance(core.get_property("CPU", props.available_devices), list)
    assert isinstance(core.get_property("CPU", props.optimal_number_of_infer_requests), int)
    assert isinstance(core.get_property("CPU", props.range_for_streams), tuple)
    assert isinstance(core.get_property("CPU", props.range_for_async_infer_requests), tuple)
    assert isinstance(core.get_property("CPU", device.full_name), str)
    assert isinstance(core.get_property("CPU", device.capabilities), list)
