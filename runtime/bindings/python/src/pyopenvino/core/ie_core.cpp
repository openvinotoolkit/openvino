// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/ie_core.hpp"

namespace py = pybind11;

using ConfigMap = std::map<std::string, std::string>;

std::string to_string(py::handle handle) {
    auto encodedString = PyUnicode_AsUTF8String(handle.ptr());
    return PyBytes_AsString(encodedString);
}

void regclass_Core(py::module m) {
    py::class_<ov::runtime::Core, std::shared_ptr<ov::runtime::Core>> cls(m, "Core");
    cls.def(py::init());
    cls.def(py::init<const std::string&>());

    cls.def(
        "set_config",
        [](ov::runtime::Core& self, const py::dict& config, const std::string& device_name) {
            std::map<std::string, std::string> config_map;
            for (auto item : config) {
                config_map[to_string(item.first)] = to_string(item.second);
            }
            self.set_config(config_map, device_name);
        },
        py::arg("config"),
        py::arg("device_name"));

    cls.def(
        "compile_model",
        (ov::runtime::ExecutableNetwork(
            ov::runtime::Core::*)(const std::shared_ptr<const ov::Function>&, const std::string&, const ConfigMap&)) &
            ov::runtime::Core::compile_model,
        // overload_cast_<const std::shared_ptr<const ov::Function>&, const std::string&, const
        // ConfigMap&>(&ov::runtime::Core::compile_model),
        py::arg("network"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def(
        "add_extension",
        [](ov::runtime::Core& self, const std::string& extension_path) {
            auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(extension_path);
            auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
            self.add_extension(extension);
        },
        py::arg("extension_path"));

    cls.def("get_versions",
            (std::map<std::string, ov::ie::Version>(ov::runtime::Core::*)(const std::string&)) &
                ov::runtime::Core::get_versions,
            py::arg("device_name"));

    cls.def(
        "read_model",
        [](ov::runtime::Core& self, py::bytes model, py::bytes weights) {
            InferenceEngine::MemoryBlob::Ptr weights_blob;
            if (weights) {
                std::string weights_bytes = weights;
                uint8_t* bin = (uint8_t*)weights_bytes.c_str();
                size_t bin_size = weights_bytes.length();
                InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8,
                                                       {bin_size},
                                                       InferenceEngine::Layout::C);
                weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
                weights_blob->allocate();
                memcpy(weights_blob->rwmap().as<uint8_t*>(), bin, bin_size);
            }
            return self.read_model(model, weights_blob);
        },
        py::arg("model"),
        py::arg("weights"));

    cls.def(
        "read_model",
        [](ov::runtime::Core& self, const std::string& model, const std::string& weights) {
            return self.read_model(model, weights);
        },
        py::arg("model"),
        py::arg("weights") = "");

    cls.def(
        "read_model",
        [](ov::runtime::Core& self, const std::string& model, py::handle blob) {
            return self.read_model(model, Common::cast_to_blob(blob));
        },
        py::arg("model"),
        py::arg("blob"));

    cls.def(
        "read_model",
        [](ov::runtime::Core& self, py::object model, py::object weights) {
            return self.read_model(py::str(model), py::str(weights));
        },
        py::arg("model"),
        py::arg("weights") = "");

    cls.def(
        "import_model",
        (ov::runtime::ExecutableNetwork(ov::runtime::Core::*)(std::istream&, const std::string&, const ConfigMap&)) &
            ov::runtime::Core::import_model,
        py::arg("model_file"),
        py::arg("device_name"),
        py::arg("config") = py::none());

    cls.def(
        "get_config",
        [](ov::runtime::Core& self, const std::string& device_name, const std::string& config_name) -> py::handle {
            return Common::parse_parameter(self.get_config(device_name, config_name));
        },
        py::arg("device_name"),
        py::arg("config_name"));

    cls.def(
        "get_metric",
        [](ov::runtime::Core& self, std::string device_name, std::string metric_name) -> py::handle {
            return Common::parse_parameter(self.get_metric(device_name, metric_name));
        },
        py::arg("device_name"),
        py::arg("metric_name"));

    cls.def("register_plugin",
            &ov::runtime::Core::register_plugin,
            py::arg("plugin_name"),
            py::arg("device_name") = py::str());

    cls.def("register_plugins", &ov::runtime::Core::register_plugins);

    cls.def("unload_plugin", &ov::runtime::Core::unload_plugin, py::arg("device_name"));

    cls.def(
        "query_model",
        (ov::runtime::SupportedOpsMap(
            ov::runtime::Core::*)(const std::shared_ptr<const ov::Function>&, const std::string&, const ConfigMap&)) &
            ov::runtime::Core::query_model,
        py::arg("model"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def_property_readonly("available_devices", &ov::runtime::Core::get_available_devices);
}
