// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/properties/properties.hpp"

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/any.hpp"

namespace py = pybind11;

// "base class" class PyProperty

void regmodule_properties(py::module m) {
    // Top submodule
    py::module m_properties = m.def_submodule("properties", "openvino.runtime.properties submodule");

    // Submodule properties - enums
    py::enum_<ov::Affinity>(m_properties, "Affinity", py::arithmetic())
        .value("NONE", ov::Affinity::NONE)
        .value("CORE", ov::Affinity::CORE)
        .value("NUMA", ov::Affinity::NUMA)
        .value("HYBRID_AWARE", ov::Affinity::HYBRID_AWARE);

    // Submodule properties - properties
    wrap_property_RW(m_properties, ov::enable_profiling, "enable_profiling");
    wrap_property_RW(m_properties, ov::cache_dir, "cache_dir");
    wrap_property_RW(m_properties, ov::auto_batch_timeout, "auto_batch_timeout");
    wrap_property_RW(m_properties, ov::num_streams, "num_streams");
    wrap_property_RW(m_properties, ov::inference_num_threads, "inference_num_threads");
    wrap_property_RW(m_properties, ov::compilation_num_threads, "compilation_num_threads");
    wrap_property_RW(m_properties, ov::affinity, "affinity");

    wrap_property_RO(m_properties, ov::supported_properties, "supported_properties");
    wrap_property_RO(m_properties, ov::available_devices, "available_devices");
    wrap_property_RO(m_properties, ov::model_name, "model_name");
    wrap_property_RO(m_properties, ov::optimal_number_of_infer_requests, "optimal_number_of_infer_requests");
    wrap_property_RO(m_properties, ov::range_for_streams, "range_for_streams");
    wrap_property_RO(m_properties, ov::optimal_batch_size, "optimal_batch_size");
    wrap_property_RO(m_properties, ov::max_batch_size, "max_batch_size");
    wrap_property_RO(m_properties, ov::range_for_async_infer_requests, "range_for_async_infer_requests");

    // Submodule hint
    py::module m_hint =
        m_properties.def_submodule("hint", "openvino.runtime.properties.hint submodule that simulates ov::hint");

    // Submodule hint - enums
    py::enum_<ov::hint::Priority>(m_hint, "Priority", py::arithmetic())
        .value("LOW", ov::hint::Priority::LOW)
        .value("MEDIUM", ov::hint::Priority::MEDIUM)
        .value("HIGH", ov::hint::Priority::HIGH)
        .value("DEFAULT", ov::hint::Priority::DEFAULT);

    py::enum_<ov::hint::PerformanceMode>(m_hint, "PerformanceMode", py::arithmetic())
        .value("UNDEFINED", ov::hint::PerformanceMode::UNDEFINED)
        .value("LATENCY", ov::hint::PerformanceMode::LATENCY)
        .value("THROUGHPUT", ov::hint::PerformanceMode::THROUGHPUT)
        .value("CUMULATIVE_THROUGHPUT", ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);

    // Submodule hint - properties
    wrap_property_RW(m_hint, ov::hint::inference_precision, "inference_precision");
    wrap_property_RW(m_hint, ov::hint::model_priority, "model_priority");
    wrap_property_RW(m_hint, ov::hint::performance_mode, "performance_mode");
    wrap_property_RW(m_hint, ov::hint::num_requests, "num_requests");
    wrap_property_RW(m_hint, ov::hint::model, "model");
    wrap_property_RW(m_hint, ov::hint::allow_auto_batching, "allow_auto_batching");

    // Submodule device
    py::module m_device =
        m_properties.def_submodule("device", "openvino.runtime.properties.device submodule that simulates ov::device");

    // Submodule device - enums
    py::enum_<ov::device::Type>(m_device, "Type", py::arithmetic())
        .value("INTEGRATED", ov::device::Type::INTEGRATED)
        .value("DISCRETE", ov::device::Type::DISCRETE);

    // Submodule device - properties
    wrap_property_RW(m_device, ov::device::id, "id");

    wrap_property_RO(m_device, ov::device::full_name, "full_name");
    wrap_property_RO(m_device, ov::device::architecture, "architecture");
    wrap_property_RO(m_device, ov::device::type, "type");
    wrap_property_RO(m_device, ov::device::gops, "gops");
    wrap_property_RO(m_device, ov::device::thermal, "thermal");
    wrap_property_RO(m_device, ov::device::capabilities, "capabilities");

    // TODO: bind struct Properties and following static constexpr Properties properties somehow...

    // Modules made in pybind cannot easily register attributes, thus workaround is needed.
    // Let's simulate module with attributes by creating empty proxy class called FakeModuleName.
    class FakeCapability {};

    py::class_<FakeCapability, std::shared_ptr<FakeCapability>> m_capability(
        m_device,
        "Capability",
        "openvino.runtime.properties.device.Capability that simulates ov::device::capability");

    m_capability.attr("FP32") = ov::device::capability::FP32;
    m_capability.attr("BF16") = ov::device::capability::BF16;
    m_capability.attr("FP16") = ov::device::capability::FP16;
    m_capability.attr("INT8") = ov::device::capability::INT8;
    m_capability.attr("INT16") = ov::device::capability::INT16;
    m_capability.attr("BIN") = ov::device::capability::BIN;
    m_capability.attr("WINOGRAD") = ov::device::capability::WINOGRAD;
    m_capability.attr("EXPORT_IMPORT") = ov::device::capability::EXPORT_IMPORT;

    // Submodule log
    py::module m_log =
        m_properties.def_submodule("log", "openvino.runtime.properties.log submodule that simulates ov::log");

    // Submodule log - enums
    py::enum_<ov::log::Level>(m_log, "Level", py::arithmetic())
        .value("NO", ov::log::Level::NO)
        .value("ERR", ov::log::Level::ERR)
        .value("WARNING", ov::log::Level::WARNING)
        .value("INFO", ov::log::Level::INFO)
        .value("DEBUG", ov::log::Level::DEBUG)
        .value("TRACE", ov::log::Level::TRACE);

    // Submodule log - properties
    wrap_property_RW(m_log, ov::log::level, "level");

    // Submodule streams
    py::module m_streams =
        m_properties.def_submodule("streams",
                                   "openvino.runtime.properties.streams submodule that simulates ov::streams");

    // TODO: struct Num which is just a tuple...

    // Submodule streams - properties RW
    wrap_property_RW(m_streams, ov::streams::num, "num");

    // TODO: static constexpr Num AUTO{-1};
    // TODO: static constexpr Num NUMA{-2};
}
