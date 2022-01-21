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
    py::class_<ov::Core, std::shared_ptr<ov::Core>> cls(m, "Core");

    cls.def(py::init<const std::string&>(), py::arg("xml_config_file") = "");

    cls.def(
        "set_config",
        [](ov::Core& self, const std::map<std::string, std::string>& config, const std::string& device_name) {
            self.set_property(device_name, {config.begin(), config.end()});
        },
        py::arg("config"),
        py::arg("device_name") = "");

    cls.def(
        "compile_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::string& device_name,
           const std::map<std::string, std::string>& config) {
            return self.compile_model(model, device_name, {config.begin(), config.end()});
        },
        py::arg("model"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::map<std::string, std::string>& config) {
            return self.compile_model(model, ov::AnyMap{config.begin(), config.end()});
        },
        py::arg("model"),
        py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        [](ov::Core& self,
           const std::string& model_path,
           const std::string& device_name,
           const std::map<std::string, std::string>& config) {
            return self.compile_model(model_path, device_name, {config.begin(), config.end()});
        },
        py::arg("model_path"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        [](ov::Core& self, const std::string& model_path, const std::map<std::string, std::string>& config) {
            return self.compile_model(model_path, ov::AnyMap{config.begin(), config.end()});
        },
        py::arg("model_path"),
        py::arg("config") = py::dict());

    cls.def("get_versions", &ov::Core::get_versions, py::arg("device_name"));

    cls.def(
        "read_model",
        [](ov::Core& self, py::bytes model, py::bytes weights) {
            // works on view in order to omit copying bytes into string
            py::buffer_info info(py::buffer(weights).request());
            size_t bin_size = static_cast<size_t>(info.size);
            // if weights are not empty
            if (bin_size) {
                const uint8_t* bin = reinterpret_cast<const uint8_t*>(info.ptr);
                ov::Tensor tensor(ov::element::Type_t::u8, {bin_size});
                std::memcpy(tensor.data(), bin, bin_size);
                return self.read_model(model, tensor);
            }
            // create empty tensor of type u8
            ov::Tensor tensor(ov::element::Type_t::u8, {});
            return self.read_model(model, tensor);
        },
        py::arg("model"),
        py::arg("weights") = py::bytes());

    cls.def(
        "read_model",
        (std::shared_ptr<ov::Model>(ov::Core::*)(const std::string&, const std::string&) const) & ov::Core::read_model,
        py::arg("model"),
        py::arg("weights") = "");

    cls.def(
        "read_model",
        (std::shared_ptr<ov::Model>(ov::Core::*)(const std::string&, const ov::Tensor&) const) & ov::Core::read_model,
        py::arg("model"),
        py::arg("weights"));

    cls.def(
        "read_model",
        [](ov::Core& self, py::object model, py::object weights) {
            return self.read_model(py::str(model), py::str(weights));
        },
        py::arg("model"),
        py::arg("weights") = "");

    cls.def(
        "import_model",
        [](ov::Core& self,
           std::istream& model_file,
           const std::string& device_name,
           const std::map<std::string, std::string>& config) {
            return self.import_model(model_file, device_name, {config.begin(), config.end()});
        },
        py::arg("model_file"),
        py::arg("device_name"),
        py::arg("config") = py::none());

    cls.def(
        "get_config",
        [](ov::Core& self, const std::string& device_name, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_property(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def(
        "get_metric",
        [](ov::Core& self, const std::string device_name, const std::string name) -> py::object {
            return Common::from_ov_any(self.get_property(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def("register_plugin", &ov::Core::register_plugin, py::arg("plugin_name"), py::arg("device_name"));

    cls.def("register_plugins", &ov::Core::register_plugins, py::arg("xml_config_file"));

    cls.def("unload_plugin", &ov::Core::unload_plugin, py::arg("device_name"));

    cls.def(
        "query_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::string& device_name,
           const std::map<std::string, std::string>& config) {
            return self.query_model(model, device_name, {config.begin(), config.end()});
        },
        py::arg("model"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def("add_extension",
            static_cast<void (ov::Core::*)(const std::string&)>(&ov::Core::add_extension),
            py::arg("library_path"));

    cls.def("add_extension",
            static_cast<void (ov::Core::*)(const std::shared_ptr<ov::Extension>&)>(&ov::Core::add_extension),
            py::arg("extension"));

    cls.def(
        "add_extension",
        static_cast<void (ov::Core::*)(const std::vector<std::shared_ptr<ov::Extension>>&)>(&ov::Core::add_extension),
        py::arg("extensions"));

    cls.def_property_readonly("available_devices", &ov::Core::get_available_devices);
}
