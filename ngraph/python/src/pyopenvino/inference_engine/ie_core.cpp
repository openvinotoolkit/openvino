// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>

#include "pyopenvino/inference_engine/ie_core.hpp"
#include <pybind11/stl.h>

#include "common.hpp"

namespace py = pybind11;

std::string to_string(py::handle handle) {
    auto encodedString = PyUnicode_AsUTF8String(handle.ptr());
    return PyBytes_AsString(encodedString);

}

void regclass_IECore(py::module m)
{
    py::class_<InferenceEngine::Core, std::shared_ptr<InferenceEngine::Core>> cls(m, "IECore");
    cls.def(py::init());
    cls.def(py::init<const std::string&>());

    cls.def("set_config", [](InferenceEngine::Core& self,
                             const py::dict& config,
                             const std::string& device_name) {
        std::map <std::string, std::string> config_map;
        for (auto item : config) {
            config_map[to_string(item.first)] = to_string(item.second);
        }
        self.SetConfig(config_map, device_name);
    }, py::arg("config"), py::arg("device_name"));

    cls.def("load_network", [](InferenceEngine::Core& self,
                               const InferenceEngine::CNNNetwork& network,
                               const std::string& device_name,
                               const std::map<std::string, std::string>& config) {
        return self.LoadNetwork(network, device_name, config);
    }, py::arg("network"), py::arg("device_name"), py::arg("config")=py::dict());

    cls.def("add_extension", [](InferenceEngine::Core& self,
                                const std::string& extension_path,
                                const std::string& device_name) {
        auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(extension_path);
        auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
        self.AddExtension(extension, device_name);
    }, py::arg("extension_path"), py::arg("device_name"));

    cls.def("get_versions", [](InferenceEngine::Core& self,
                               const std::string& device_name) {
        return self.GetVersions(device_name);
    }, py::arg("device_name"));

    cls.def("read_network", [](InferenceEngine::Core& self,
                               py::bytes model,
                               py::bytes weights) {
        InferenceEngine::MemoryBlob::Ptr weights_blob;
        if(weights){
            std::string weights_bytes = weights;
            uint8_t* bin = (uint8_t*)weights_bytes.c_str();
            size_t bin_size = weights_bytes.length();
            InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, { bin_size }, InferenceEngine::Layout::C);
            weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
            weights_blob->allocate();
            memcpy(weights_blob->rwmap().as<uint8_t*>(), bin, bin_size);
        }
        return self.ReadNetwork(model,weights_blob);
    }, py::arg("model"), py::arg("weights"));

    cls.def("read_network", [](InferenceEngine::Core& self,
                               const std::string& model,
                               const std::string& weights) {
        return self.ReadNetwork(model, weights);
    }, py::arg("model"), py::arg("weights")="");

    cls.def("read_network", [](InferenceEngine::Core& self,
                               const std::string& model,
                               py::handle blob) {
        return self.ReadNetwork(model, Common::cast_to_blob(blob));
    }, py::arg("model"), py::arg("blob"));

    cls.def("read_network", [](InferenceEngine::Core& self,
                               py::object model,
                               py::object weights) {
        return self.ReadNetwork(py::str(model), py::str(weights));
    }, py::arg("model"), py::arg("weights")="");

    cls.def("import_network", [](InferenceEngine::Core& self,
                                 const std::string& model_file,
                                 const std::string& device_name,
                                 const std::map<std::string, std::string>& config) {
        return self.ImportNetwork(model_file, device_name, config);
    }, py::arg("model_file"), py::arg("device_name"), py::arg("config")=py::none());

    cls.def("get_config", [](InferenceEngine::Core& self,
                             const std::string& device_name,
                             const std::string& config_name) -> py::handle {
        return Common::parse_parameter(self.GetConfig(device_name, config_name));
    }, py::arg("device_name"), py::arg("config_name"));

    cls.def("get_metric", [](InferenceEngine::Core& self,
                             std::string device_name,
                             std::string metric_name) -> py::handle {
        return Common::parse_parameter(self.GetMetric(device_name, metric_name));
    }, py::arg("device_name"), py::arg("metric_name"));

    cls.def("register_plugin", &InferenceEngine::Core::RegisterPlugin,
            py::arg("plugin_name"), py::arg("device_name")=py::str());

    cls.def("register_plugins", &InferenceEngine::Core::RegisterPlugins);

    cls.def("unregister_plugin", &InferenceEngine::Core::UnregisterPlugin, py::arg("device_name"));

    cls.def("query_network", [](InferenceEngine::Core& self,
                             const InferenceEngine::CNNNetwork& network,
                             const std::string& device_name,
                             const std::map<std::string, std::string>& config)  {
        return self.QueryNetwork(network, device_name, config).supportedLayersMap;
    }, py::arg("network"), py::arg("device_name"), py::arg("config")=py::dict());

    cls.def_property_readonly("available_devices", &InferenceEngine::Core::GetAvailableDevices);
}
