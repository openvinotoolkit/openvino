// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_api_impl.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include "ie_iinfer_request.hpp"
#include "details/ie_cnn_network_tools.h"

#include "helpers.h"

PyObject *parse_parameter(const InferenceEngine::Parameter &param) {
    // Check for std::string
    if (param.is<std::string>()) {
        return PyUnicode_FromString(param.as<std::string>().c_str());
    }
        // Check for int
    else if (param.is<int>()) {
        auto val = param.as<int>();
        return PyLong_FromLong((long) val);
    }
        // Check for unsinged int
    else if (param.is<unsigned int>()) {
        auto val = param.as<unsigned int>();
        return PyLong_FromLong((unsigned long) val);
    }
        // Check for float
    else if (param.is<float>()) {
        auto val = param.as<float>();
        return PyFloat_FromDouble((double) val);
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
        for (const auto &it : val) {
            PyObject *str_val = PyUnicode_FromString(it.c_str());
            PyList_Append(list, str_val);
        }
        return list;
    }
        // Check for std::vector<int>
    else if (param.is<std::vector<int>>()) {
        auto val = param.as<std::vector<int>>();
        PyObject *list = PyList_New(0);
        for (const auto &it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<unsigned int>
    else if (param.is<std::vector<unsigned int>>()) {
        auto val = param.as<std::vector<unsigned int>>();
        PyObject *list = PyList_New(0);
        for (const auto &it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<float>
    else if (param.is<std::vector<float>>()) {
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
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long) std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long) std::get<1>(val)));
        return tuple;
    }
        // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int, unsigned int >>();
        PyObject *tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long) std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long) std::get<1>(val)));
        PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long) std::get<2>(val)));
        return tuple;
    }
        // Check for std::map<std::string, std::string>
    else if (param.is<std::map<std::string, std::string>>()) {
        auto val = param.as<std::map<std::string, std::string>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
        }
        return dict;
    }
        // Check for std::map<std::string, int>
    else if (param.is<std::map<std::string, int>>()) {
        auto val = param.as<std::map<std::string, int>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long) it.second));
        }
        return dict;
    } else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return (PyObject *) NULL;
    }
}

InferenceEnginePython::IEExecNetworkPython::IEExecNetworkPython(const std::string &name, size_t num_requests) :
        IEExecNetwork(name, num_requests) {
}

PyObject *InferenceEnginePython::IEExecNetworkPython::getMetric(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(this->exec_network_ptr->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

PyObject *InferenceEnginePython::IEExecNetworkPython::getConfig(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(this->exec_network_ptr->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

InferenceEnginePython::IECorePython::IECorePython(const std::string &xmlConfigFile) :
        IECore(xmlConfigFile) {
}

PyObject *InferenceEnginePython::IECorePython::getMetric(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetMetric(deviceName, name);
    return parse_parameter(param);
}

PyObject *InferenceEnginePython::IECorePython::getConfig(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetConfig(deviceName, name);
    return parse_parameter(param);
}
