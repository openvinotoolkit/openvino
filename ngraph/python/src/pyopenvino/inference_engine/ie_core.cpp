//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ie_core.hpp>

#include "pyopenvino/inference_engine/ie_core.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;

std::string to_string(py::handle handle) {
    auto encodedString = PyUnicode_AsUTF8String(handle.ptr());
    return PyBytes_AsString(encodedString);

}


PyObject *parse_parameter(const InferenceEngine::Parameter &param) {
    // Check for std::string
    if (param.is<std::string>()) {
        return PyUnicode_FromString(param.as<std::string>().c_str());
    }
        // Check for int
    else if (param.is<int>()) {
        auto val = param.as<int>();
        return PyLong_FromLong((long)val);
    }
        // Check for unsinged int
    else if (param.is<unsigned int>()) {
        auto val = param.as<unsigned int>();
        return PyLong_FromLong((unsigned long)val);
    }
        // Check for float
    else if (param.is<float>()) {
        auto val = param.as<float>();
        return PyFloat_FromDouble((double)val);
    }
        // Check for bool
    else if (param.is<bool>()) {
        auto val = param.as<bool>();
        return val ? Py_True : Py_False;
    }
        // Check for std::vector<std::string>
    else if (param.is<std::vector<std::string>>()) {
        auto val = param.as<std::vector<std::string>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyObject *str_val = PyUnicode_FromString(it.c_str());
            PyList_Append(list, str_val);
        }
        return list;
    }
        // Check for std::vector<int>
    else if (param.is<std::vector<int>>()){
        auto val = param.as<std::vector<int>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<unsigned int>
    else if (param.is<std::vector<unsigned int>>()){
        auto val = param.as<std::vector<unsigned int>>();
        PyObject *list = PyList_New(0);
        for (const auto &it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<float>
    else if (param.is<std::vector<float>>()){
        auto val = param.as<std::vector<float>>();
        PyObject *list = PyList_New(0);
        for (const auto &it : val) {
            PyList_Append(list, PyFloat_FromDouble((double) it));
        }
        return list;
    }
        // Check for std::tuple<unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int >>();
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        return tuple;
    }
        // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int, unsigned int >>();
        PyObject *tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long)std::get<2>(val)));
        return tuple;
    }
        // Check for std::map<std::string, std::string>
    else if (param.is<std::map<std::string, std::string>>()) {
        auto val = param.as<std::map<std::string, std::string>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val){
            PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
        }
        return dict;
    }
        // Check for std::map<std::string, int>
    else if (param.is<std::map<std::string, int>>()) {
        auto val = param.as<std::map<std::string, int>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val){
            PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long)it.second));
        }
        return dict;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return (PyObject *) NULL;
    }
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
                               const std::string& model,
                               const std::string& weights) {
        return self.ReadNetwork(model, weights);
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
        return parse_parameter(self.GetConfig(device_name, config_name));
    }, py::arg("device_name"), py::arg("config_name"));

    cls.def("get_metric", [](InferenceEngine::Core& self,
                             std::string device_name,
                             std::string metric_name) -> py::handle {
        return parse_parameter(self.GetMetric(device_name, metric_name));
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
