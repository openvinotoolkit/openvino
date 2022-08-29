// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/intel_gna/properties.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/intel_myriad/hddl_properties.hpp>
#include <openvino/runtime/intel_myriad/myriad_properties.hpp>

#define VALUE_2_STRING(name) #name
#define INSERT_NAME_VALUE_MAP(out_map, value)        \
    {                                                \
        std::string str_key = VALUE_2_STRING(value); \
        std::stringstream str_value;                 \
        str_value << value;                          \
        out_map[str_key] = str_value.str();          \
    }

enum PropertyType {
    PROPERTY_TYPE_STRING = 0,
    PROPERTY_TYPE_INT,
    PROPERTY_TYPE_UNSIGNED_INT,
    PROPERTY_TYPE_INT64,
    PROPERTY_TYPE_ENUM,
    PROPERTY_TYPE_BOOL,
    PROPERTY_TYPE_FLOAT,
    PROPERTY_TYPE_STRING_FLOAT_MAP,
};

struct PropertyInfo {
    std::string property_name;
    PropertyType property_type;
};

#define INSERT_PROPERTY_MAP(out_map, value, type, reversed) \
    {                                                       \
        std::string str_key = VALUE_2_STRING(value);        \
        std::string str_name = value.name();                \
        PropertyInfo property_info;                         \
        property_info.property_type = type;                 \
        if (reversed) {                                     \
            property_info.property_name = str_key;          \
            out_map[str_name] = property_info;              \
        } else {                                            \
            property_info.property_name = str_name;         \
            out_map[str_key] = property_info;               \
        }                                                   \
    }

void generate_property_map(std::map<std::string, PropertyInfo>& property_map, bool reversed) {
    INSERT_PROPERTY_MAP(property_map, ov::enable_profiling, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::cache_dir, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::auto_batch_timeout, PROPERTY_TYPE_UNSIGNED_INT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::num_streams, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::inference_num_threads, PROPERTY_TYPE_INT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::compilation_num_threads, PROPERTY_TYPE_INT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::affinity, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::device::id, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::device::priorities, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::hint::inference_precision, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::hint::model_priority, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::hint::performance_mode, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::hint::allow_auto_batching, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::hint::num_requests, PROPERTY_TYPE_UNSIGNED_INT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::log::level, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::streams::num, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::force_tbb_terminate, PROPERTY_TYPE_BOOL, reversed);

    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::enable_force_reset, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::protocol, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::ddr_type, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::enable_hw_acceleration, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::enable_receiving_tensor_time, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::custom_layers, PROPERTY_TYPE_STRING, reversed);

    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::firmware_model_image_path, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::pwl_max_error_percent, PROPERTY_TYPE_FLOAT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::memory_reuse, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::execution_mode, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::execution_target, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::compile_target, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::pwl_design_algorithm, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gna::scale_factors_per_input, PROPERTY_TYPE_STRING_FLOAT_MAP, reversed);

    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::context_type, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::ocl_context_device_id, PROPERTY_TYPE_INT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::tile_id, PROPERTY_TYPE_INT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::shared_mem_type, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::va_plane, PROPERTY_TYPE_UNSIGNED_INT, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::enable_loop_unrolling, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::hint::queue_throttle, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::hint::queue_priority, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::hint::host_task_priority, PROPERTY_TYPE_ENUM, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_gpu::hint::available_device_mem, PROPERTY_TYPE_INT64, reversed);

    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::hddl::graph_tag, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::hddl::set_stream_id, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::hddl::set_device_tag, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::hddl::bind_device, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::hddl::runtime_priority, PROPERTY_TYPE_STRING, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::hddl::use_sgad, PROPERTY_TYPE_BOOL, reversed);
    INSERT_PROPERTY_MAP(property_map, ov::intel_myriad::hddl::group_device, PROPERTY_TYPE_STRING, reversed);
}

void generate_enum_map(std::map<std::string, std::string>& enum_map) {
    INSERT_NAME_VALUE_MAP(enum_map, ov::Affinity::NONE);
    INSERT_NAME_VALUE_MAP(enum_map, ov::Affinity::CORE);
    INSERT_NAME_VALUE_MAP(enum_map, ov::Affinity::NUMA);
    INSERT_NAME_VALUE_MAP(enum_map, ov::Affinity::HYBRID_AWARE);

    INSERT_NAME_VALUE_MAP(enum_map, ov::device::Type::INTEGRATED);
    INSERT_NAME_VALUE_MAP(enum_map, ov::device::Type::DISCRETE);

    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::Priority::LOW);
    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::Priority::MEDIUM);
    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::Priority::HIGH);
    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::Priority::DEFAULT);

    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::PerformanceMode::UNDEFINED);
    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::PerformanceMode::LATENCY);
    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::PerformanceMode::THROUGHPUT);
    INSERT_NAME_VALUE_MAP(enum_map, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);

    INSERT_NAME_VALUE_MAP(enum_map, ov::log::Level::NO);
    INSERT_NAME_VALUE_MAP(enum_map, ov::log::Level::ERR);
    INSERT_NAME_VALUE_MAP(enum_map, ov::log::Level::WARNING);
    INSERT_NAME_VALUE_MAP(enum_map, ov::log::Level::INFO);
    INSERT_NAME_VALUE_MAP(enum_map, ov::log::Level::DEBUG);
    INSERT_NAME_VALUE_MAP(enum_map, ov::log::Level::TRACE);

    INSERT_NAME_VALUE_MAP(enum_map, ov::element::bf16);
    INSERT_NAME_VALUE_MAP(enum_map, ov::element::f32);

    INSERT_NAME_VALUE_MAP(enum_map, ov::streams::AUTO);
    INSERT_NAME_VALUE_MAP(enum_map, ov::streams::NUMA);

    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_myriad::Protocol::USB);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_myriad::Protocol::PCIE);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_myriad::DDRType::MYRIAD_DDR_AUTO);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_myriad::DDRType::MYRIAD_DDR_MICRON_2GB);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_myriad::DDRType::MYRIAD_DDR_SAMSUNG_2GB);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_myriad::DDRType::MYRIAD_DDR_HYNIX_2GB);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_myriad::DDRType::MYRIAD_DDR_MICRON_1GB);

    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::ExecutionMode::AUTO);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::ExecutionMode::HW);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::ExecutionMode::SW_EXACT);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::ExecutionMode::SW_FP32);

    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::HWGeneration::UNDEFINED);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::HWGeneration::GNA_2_0);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::HWGeneration::GNA_3_0);

    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::PWLDesignAlgorithm::UNDEFINED);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::PWLDesignAlgorithm::RECURSIVE_DESCENT);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gna::PWLDesignAlgorithm::UNIFORM_DISTRIBUTION);

    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::ContextType::OCL);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::ContextType::VA_SHARED);

    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::SharedMemType::OCL_BUFFER);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::SharedMemType::OCL_IMAGE2D);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::SharedMemType::USM_USER_BUFFER);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::SharedMemType::USM_HOST_BUFFER);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::SharedMemType::VA_SURFACE);
    INSERT_NAME_VALUE_MAP(enum_map, ov::intel_gpu::SharedMemType::DX_BUFFER);
}
