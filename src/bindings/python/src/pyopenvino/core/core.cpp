// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/core.hpp"

#include <ie_extension.h>
#include <pybind11/stl.h>

#include <openvino/core/any.hpp>
#include <openvino/runtime/core.hpp>
#include <pyopenvino/core/tensor.hpp>

#include "common.hpp"

namespace py = pybind11;

using ConfigMap = std::map<std::string, std::string>;

std::string to_string(py::handle handle) {
    auto encodedString = PyUnicode_AsUTF8String(handle.ptr());
    return PyBytes_AsString(encodedString);
}

void regclass_Core(py::module m) {
    py::class_<ov::runtime::Core, std::shared_ptr<ov::runtime::Core>> cls(m, "Core");

    cls.def(py::init<const std::string&>(), py::arg("xml_config_file") = "");

    cls.def("set_config",
            (void (ov::runtime::Core::*)(const ConfigMap&, const std::string&)) & ov::runtime::Core::set_config,
            py::arg("config"),
            py::arg("device_name") = "");

    cls.def("compile_model",
            (ov::runtime::CompiledModel(
                ov::runtime::Core::*)(const std::shared_ptr<const ov::Model>&, const std::string&, const ConfigMap&)) &
                ov::runtime::Core::compile_model,
            py::arg("model"),
            py::arg("device_name"),
            py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        (ov::runtime::CompiledModel(ov::runtime::Core::*)(const std::shared_ptr<const ov::Model>&, const ConfigMap&)) &
            ov::runtime::Core::compile_model,
        py::arg("model"),
        py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        (ov::runtime::CompiledModel(ov::runtime::Core::*)(const std::string&, const std::string&, const ConfigMap&)) &
            ov::runtime::Core::compile_model,
        py::arg("model_path"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def("compile_model",
            (ov::runtime::CompiledModel(ov::runtime::Core::*)(const std::string&, const ConfigMap&)) &
                ov::runtime::Core::compile_model,
            py::arg("model_path"),
            py::arg("config") = py::dict());

    cls.def("get_versions", &ov::runtime::Core::get_versions, py::arg("device_name"));

    cls.def(
        "read_model",
        [](ov::runtime::Core& self, py::bytes model, py::bytes weights) {
            // works on view in order to omit copying bytes into string
            py::buffer_info info(py::buffer(weights).request());
            size_t bin_size = static_cast<size_t>(info.size);
            // if weights are not empty
            if (bin_size) {
                const uint8_t* bin = reinterpret_cast<const uint8_t*>(info.ptr);
                ov::runtime::Tensor tensor(ov::element::Type_t::u8, {bin_size});
                std::memcpy(tensor.data(), bin, bin_size);
                return self.read_model(model, tensor);
            }
            // create empty tensor of type u8
            ov::runtime::Tensor tensor(ov::element::Type_t::u8, {});
            return self.read_model(model, tensor);
        },
        py::arg("model"),
        py::arg("weights") = py::bytes());

    cls.def("read_model",
            (std::shared_ptr<ov::Model>(ov::runtime::Core::*)(const std::string&, const std::string&) const) &
                ov::runtime::Core::read_model,
            py::arg("model"),
            py::arg("weights") = "");

    cls.def("read_model",
            (std::shared_ptr<ov::Model>(ov::runtime::Core::*)(const std::string&, const ov::runtime::Tensor&) const) &
                ov::runtime::Core::read_model,
            py::arg("model"),
            py::arg("weights"));

    cls.def(
        "read_model",
        [](ov::runtime::Core& self, py::object model, py::object weights) {
            return self.read_model(py::str(model), py::str(weights));
        },
        py::arg("model"),
        py::arg("weights") = "");

    cls.def("import_model",
            (ov::runtime::CompiledModel(ov::runtime::Core::*)(std::istream&, const std::string&, const ConfigMap&)) &
                ov::runtime::Core::import_model,
            py::arg("model_file"),
            py::arg("device_name"),
            py::arg("config") = py::none());

    cls.def(
        "get_config",
        [](ov::runtime::Core& self, const std::string& device_name, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_config(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def(
        "get_metric",
        [](ov::runtime::Core& self, const std::string device_name, const std::string name) -> py::object {
            return Common::from_ov_any(self.get_metric(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def("register_plugin", &ov::runtime::Core::register_plugin, py::arg("plugin_name"), py::arg("device_name"));

    cls.def("register_plugins", &ov::runtime::Core::register_plugins, py::arg("xml_config_file"));

    cls.def("unload_plugin", &ov::runtime::Core::unload_plugin, py::arg("device_name"));

    cls.def("query_model",
            (ov::runtime::SupportedOpsMap(
                ov::runtime::Core::*)(const std::shared_ptr<const ov::Model>&, const std::string&, const ConfigMap&)) &
                ov::runtime::Core::query_model,
            py::arg("model"),
            py::arg("device_name"),
            py::arg("config") = py::dict());

    cls.def("add_extension",
            static_cast<void (ov::runtime::Core::*)(const std::string&)>(&ov::runtime::Core::add_extension),
            py::arg("library_path"));

    cls.def("add_extension",
            static_cast<void (ov::runtime::Core::*)(const std::shared_ptr<ov::Extension>&)>(
                &ov::runtime::Core::add_extension),
            py::arg("extension"));

    cls.def("add_extension",
            static_cast<void (ov::runtime::Core::*)(const std::vector<std::shared_ptr<ov::Extension>>&)>(
                &ov::runtime::Core::add_extension),
            py::arg("extensions"));

    cls.def_property_readonly("available_devices", &ov::runtime::Core::get_available_devices);
}
