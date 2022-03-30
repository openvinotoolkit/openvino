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

    // Property specializations...
    py::module m_specializations = m_properties.def_submodule("specializations", "openvino.runtime.properties.specializations submodule");

    // Register this ... thing ... as a base
    py::class_<ov::util::PropertyTag, std::shared_ptr<ov::util::PropertyTag>> property_base(m_specializations, "PropertyBase");

    auto property0 =
        wrap_property<ov::util::PropertyTag, std::vector<ov::PropertyName>, ov::PropertyMutability::RO>(m_specializations, "0");

    auto property1 = wrap_property<ov::util::PropertyTag, std::vector<std::string>, ov::PropertyMutability::RO>(m_specializations, "1");

    auto property2 = wrap_property<ov::util::PropertyTag, ov::element::Type, ov::PropertyMutability::RW>(m_properties, "2");

    // Base things
    // Single if "RO" 
    m_properties.def("supported_properties", []() {
        return ov::supported_properties;
    });

    m_properties.def("available_devices", []() {
        return ov::available_devices;
    });

    // Submodules of openvino.runtime.properties
    // Submodule hint
    py::module m_hint =
        m_properties.def_submodule("hint", "openvino.runtime.properties.hint submodule that simulates ov::hint");

    // Pair if "RW"
    m_hint.def("inference_precision", []() {
        return ov::hint::inference_precision;
    });

    m_hint.def("inference_precision", [](ov::element::Type ov_type) {
        // TODO: cast to dict somehow?
        auto pair = ov::hint::inference_precision(ov_type);
        return std::pair<std::string, PyAny>(pair.first, Common::from_ov_any(pair.second));
    });

    // Submodule device
    py::module m_device =
        m_properties.def_submodule("device", "openvino.runtime.properties.device submodule that simulates ov::device");

    // Submodule device.capability
    py::module m_capability = m_device.def_submodule(
        "capability",
        "openvino.runtime.properties.device.capability submodule that simulates ov::device::capability");

    // Submodule log
    py::module m_log =
        m_properties.def_submodule("log", "openvino.runtime.properties.log submodule that simulates ov::log");

    // Submodule streams
    py::module m_streams =
        m_properties.def_submodule("streams",
                                   "openvino.runtime.properties.streams submodule that simulates ov::streams");
}
