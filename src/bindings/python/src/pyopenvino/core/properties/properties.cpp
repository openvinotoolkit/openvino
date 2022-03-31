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

    // Property specializations
    py::module m_specializations = m_properties.def_submodule("specializations", "openvino.runtime.properties.specializations submodule");

    // Register a proxy base
    py::class_<ov::util::PropertyTag, std::shared_ptr<ov::util::PropertyTag>> property_base(m_specializations, "PropertyBase");

    register_property_class<std::vector<ov::PropertyName>, ov::PropertyMutability::RO>(m_specializations, "VectorPropertyName");

    register_property_class<std::vector<std::string>, ov::PropertyMutability::RO>(m_specializations, "VectorString");
    
    register_property_class<std::string, ov::PropertyMutability::RO>(m_specializations, "String");

    register_property_class<uint32_t, ov::PropertyMutability::RO>(m_specializations, "Uint32");

    register_property_class<ov::element::Type, ov::PropertyMutability::RW>(m_specializations, "ElementType");

    register_property_class<ov::hint::Priority, ov::PropertyMutability::RW>(m_specializations, "Priority");

    register_property_class<ov::hint::PerformanceMode, ov::PropertyMutability::RW>(m_specializations, "PerformanceMode");

    register_property_class<uint32_t, ov::PropertyMutability::RW>(m_specializations, "Uint32");

    register_property_class<bool, ov::PropertyMutability::RW>(m_specializations, "Bool");

    register_property_class<ov::log::Level, ov::PropertyMutability::RW>(m_specializations, "Level");

    register_property_class<std::tuple<unsigned int, unsigned int>, ov::PropertyMutability::RO>(m_specializations, "UintUint");

    // register_property_class<unsigned int, ov::PropertyMutability::RO>(m_specializations, "Uint"); same as uint32_t

    register_property_class<std::tuple<unsigned int, unsigned int, unsigned int>, ov::PropertyMutability::RO>(m_specializations, "TupleUintUintUint");

    register_property_class<std::string, ov::PropertyMutability::RW>(m_specializations, "String");

    register_property_class<ov::device::Type, ov::PropertyMutability::RO>(m_specializations, "DeviceType");

    register_property_class<std::map<ov::element::Type, float>, ov::PropertyMutability::RO>(m_specializations, "MapElementTypeFloat");

    register_property_class<float, ov::PropertyMutability::RO>(m_specializations, "Float");

    register_property_class<ov::streams::Num, ov::PropertyMutability::RW>(m_specializations, "Num");

    register_property_class<int32_t, ov::PropertyMutability::RW>(m_specializations, "Int32");

    register_property_class<ov::Affinity, ov::PropertyMutability::RW>(m_specializations, "Affinity");
    ////

    // Base things
    // Single if "RO"
    wrap_property_RO(m_properties, ov::supported_properties, "supported_properties");
    wrap_property_RO(m_properties, ov::available_devices, "available_devices");
    wrap_property_RO(m_properties, ov::model_name, "model_name");
    wrap_property_RO(m_properties, ov::optimal_number_of_infer_requests, "optimal_number_of_infer_requests");

    wrap_property_RW(m_properties, ov::enable_profiling, "enable_profiling");
    wrap_property_RW(m_properties, ov::cache_dir, "cache_dir");

    wrap_property_RO(m_properties, ov::range_for_streams, "range_for_streams");
    wrap_property_RO(m_properties, ov::optimal_batch_size, "optimal_batch_size");
    wrap_property_RO(m_properties, ov::max_batch_size, "max_batch_size");

    wrap_property_RW(m_properties, ov::auto_batch_timeout, "auto_batch_timeout");
    wrap_property_RO(m_properties, ov::range_for_async_infer_requests, "range_for_async_infer_requests");

    wrap_property_RW(m_properties, ov::num_streams, "num_streams");
    wrap_property_RW(m_properties, ov::inference_num_threads, "inference_num_threads");
    wrap_property_RW(m_properties, ov::compilation_num_threads, "compilation_num_threads");

    py::enum_<ov::Affinity>(m_properties, "Affinity", py::arithmetic())
        .value("NONE", ov::Affinity::NONE)
        .value("CORE", ov::Affinity::CORE)
        .value("NUMA", ov::Affinity::NUMA)
        .value("HYBRID_AWARE", ov::Affinity::HYBRID_AWARE);

    wrap_property_RW(m_properties, ov::affinity, "affinity");

    // Submodules of openvino.runtime.properties
    // Submodule hint
    py::module m_hint =
        m_properties.def_submodule("hint", "openvino.runtime.properties.hint submodule that simulates ov::hint");

    py::enum_<ov::hint::Priority>(m_hint, "Priority", py::arithmetic())
        .value("LOW", ov::hint::Priority::LOW)
        .value("MEDIUM", ov::hint::Priority::MEDIUM)
        .value("HIGH", ov::hint::Priority::HIGH)
        .value("DEFAULT", ov::hint::Priority::DEFAULT);
    // do not export_values()!

    // Enum
    py::enum_<ov::hint::PerformanceMode>(m_hint, "PerformanceMode", py::arithmetic())
        .value("UNDEFINED", ov::hint::PerformanceMode::UNDEFINED)
        .value("LATENCY", ov::hint::PerformanceMode::LATENCY)
        .value("THROUGHPUT", ov::hint::PerformanceMode::THROUGHPUT)
        .value("CUMULATIVE_THROUGHPUT", ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);
    // do not export_values()!

    wrap_property_RW(m_hint, ov::hint::inference_precision, "inference_precision");
    wrap_property_RW(m_hint, ov::hint::model_priority, "model_priority");
    wrap_property_RW(m_hint, ov::hint::performance_mode, "performance_mode");
    wrap_property_RW(m_hint, ov::hint::num_requests, "num_requests");
    // TODO: check MODEL_PTR
    wrap_property_RW(m_hint, ov::hint::model, "model");
    wrap_property_RW(m_hint, ov::hint::allow_auto_batching, "allow_auto_batching");

    // Submodule device
    py::module m_device =
        m_properties.def_submodule("device", "openvino.runtime.properties.device submodule that simulates ov::device");

    // TODO: "id" is reserved name in python?
    wrap_property_RW(m_device, ov::device::id, "device_id");

    wrap_property_RO(m_device, ov::device::full_name, "full_name");
    wrap_property_RO(m_device, ov::device::architecture, "architecture");
    // TODO: bind struct Properties and following static constexpr Properties properties somehow...

    py::enum_<ov::device::Type>(m_device, "Type", py::arithmetic())
        .value("INTEGRATED", ov::device::Type::INTEGRATED)
        .value("DISCRETE", ov::device::Type::DISCRETE);

    // TODO: another keyword?
    wrap_property_RO(m_device, ov::device::type, "type");
    wrap_property_RO(m_device, ov::device::gops, "gops");
    wrap_property_RO(m_device, ov::device::thermal, "thermal");
    wrap_property_RO(m_device, ov::device::capabilities, "capabilities");

    // Submodule device.capability
    // py::module m_capability = m_device.def_submodule(
    //     "capability",
    //     "openvino.runtime.properties.device.capability submodule that simulates ov::device::capability");

    // TODO: it should be class with attr in my opinion, so it matches calling
    class FakeCapability {};

    py::class_<FakeCapability, std::shared_ptr<FakeCapability>> m_capability(m_device, "Capability");

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

    py::enum_<ov::log::Level>(m_log, "Level", py::arithmetic())
        .value("NO", ov::log::Level::NO)
        .value("ERR", ov::log::Level::ERR)
        .value("WARNING", ov::log::Level::WARNING)
        .value("INFO", ov::log::Level::INFO)
        .value("DEBUG", ov::log::Level::DEBUG)
        .value("TRACE", ov::log::Level::TRACE);

    wrap_property_RW(m_log, ov::log::level, "allow_auto_batching");

    // Submodule streams
    py::module m_streams =
        m_properties.def_submodule("streams",
                                   "openvino.runtime.properties.streams submodule that simulates ov::streams");

    // TODO: struct Num which is just a tuple...

    wrap_property_RW(m_streams, ov::streams::num, "num");

    // TODO: static constexpr Num AUTO{-1};
    // TODO: static constexpr Num NUMA{-2};

}
