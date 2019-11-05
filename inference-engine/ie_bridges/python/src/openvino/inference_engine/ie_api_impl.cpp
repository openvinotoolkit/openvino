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

InferenceEnginePython::IEExecNetwork::IEExecNetwork(const std::string &name, size_t num_requests) :
        InferenceEngineBridge::IEExecNetwork(name, num_requests) {
}

PyObject *InferenceEnginePython::IEExecNetwork::getMetric(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(this->exec_network_ptr->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

PyObject *InferenceEnginePython::IEExecNetwork::getConfig(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(this->exec_network_ptr->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

InferenceEnginePython::IECore::IECore(const std::string &xmlConfigFile) :
        InferenceEngineBridge::IECore(xmlConfigFile) {
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork>
InferenceEnginePython::IECore::loadNetwork(InferenceEngineBridge::IENetwork network,
                                           const std::string &deviceName,
                                           const std::map<std::string, std::string> &config,
                                           int &num_requests) {

    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEngineBridge::make_unique<InferenceEnginePython::IEExecNetwork>(network.name,
                                                                                                 num_requests);
    exec_network->exec_network_ptr = actual.LoadNetwork(network.actual, deviceName, config);

    if (0 == num_requests) {
        num_requests = InferenceEngineBridge::getOptimalNumberOfRequests(exec_network->exec_network_ptr);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferenceEngineBridge::InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->exec_network_ptr->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}

PyObject *InferenceEnginePython::IECore::getMetric(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetMetric(deviceName, name);
    return parse_parameter(param);
}

PyObject *InferenceEnginePython::IECore::getConfig(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetConfig(deviceName, name);
    return parse_parameter(param);
}

InferenceEnginePython::IEPlugin::IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs) :
        InferenceEngineBridge::IEPlugin(device, plugin_dirs) {}

std::unique_ptr<InferenceEnginePython::IEExecNetwork>
InferenceEnginePython::IEPlugin::load(const InferenceEngineBridge::IENetwork &net,
                                      int num_requests,
                                      const std::map<std::string, std::string> &config) {
    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEngineBridge::make_unique<InferenceEnginePython::IEExecNetwork>(net.name,
                                                                                                 num_requests);
    exec_network->exec_network_ptr = actual.LoadNetwork(net.actual, config);

    if (0 == num_requests) {
        num_requests = InferenceEngineBridge::getOptimalNumberOfRequests(exec_network->exec_network_ptr);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferenceEngineBridge::InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->exec_network_ptr->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}