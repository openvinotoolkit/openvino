// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <string>
#include <utility>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <tuple>
#include <memory>
#include <ie_extension.h>
#include "inference_engine.hpp"
#include "ie_compound_blob.h"
#include "c_api/ie_c_api.h"

namespace IE = InferenceEngine;

/**
 * @struct ie_core
 * @brief This struct represents Inference Engine Core entity.
 */
struct ie_core {
    IE::Core object;
};

/**
 * @struct ie_executable
 * @brief This is an interface of an executable network
 */
struct ie_executable {
    IE::ExecutableNetwork object;
};

/**
 * @struct ie_infer_request
 * @brief This is an interface of asynchronous infer request
 */
struct ie_infer_request {
    IE::InferRequest object;
};

/**
 * @struct ie_blob
 * @brief This struct represents a universal container in the Inference Engine
 */
struct ie_blob {
    IE::Blob::Ptr object;
};

/**
 * @struct ie_network
 * @brief This is the main interface to describe the NN topology
 */
struct ie_network {
    IE::CNNNetwork object;
};

std::map<IE::StatusCode, IEStatusCode> status_map = {{IE::StatusCode::GENERAL_ERROR, IEStatusCode::GENERAL_ERROR},
                                                        {IE::StatusCode::INFER_NOT_STARTED, IEStatusCode::INFER_NOT_STARTED},
                                                        {IE::StatusCode::NETWORK_NOT_LOADED,  IEStatusCode::NETWORK_NOT_LOADED},
                                                        {IE::StatusCode::NETWORK_NOT_READ,  IEStatusCode::NETWORK_NOT_READ},
                                                        {IE::StatusCode::NOT_ALLOCATED,  IEStatusCode::NOT_ALLOCATED},
                                                        {IE::StatusCode::NOT_FOUND,   IEStatusCode::NOT_FOUND},
                                                        {IE::StatusCode::NOT_IMPLEMENTED,  IEStatusCode::NOT_IMPLEMENTED},
                                                        {IE::StatusCode::OK,   IEStatusCode::OK},
                                                        {IE::StatusCode::OUT_OF_BOUNDS, IEStatusCode::OUT_OF_BOUNDS},
                                                        {IE::StatusCode::PARAMETER_MISMATCH, IEStatusCode::PARAMETER_MISMATCH},
                                                        {IE::StatusCode::REQUEST_BUSY, IEStatusCode::REQUEST_BUSY},
                                                        {IE::StatusCode::RESULT_NOT_READY, IEStatusCode::RESULT_NOT_READY},
                                                        {IE::StatusCode::UNEXPECTED, IEStatusCode::UNEXPECTED}};

std::map<IE::Precision, precision_e> precision_map = {{IE::Precision::UNSPECIFIED, precision_e::UNSPECIFIED},
                                                        {IE::Precision::MIXED, precision_e::MIXED},
                                                        {IE::Precision::FP32, precision_e::FP32},
                                                        {IE::Precision::FP16, precision_e::FP16},
                                                        {IE::Precision::FP64, precision_e::FP64},
                                                        {IE::Precision::Q78, precision_e::Q78},
                                                        {IE::Precision::I16, precision_e::I16},
                                                        {IE::Precision::U4, precision_e::U4},
                                                        {IE::Precision::U8, precision_e::U8},
                                                        {IE::Precision::I4, precision_e::I4},
                                                        {IE::Precision::I8, precision_e::I8},
                                                        {IE::Precision::U16, precision_e::U16},
                                                        {IE::Precision::I32, precision_e::I32},
                                                        {IE::Precision::U32, precision_e::U32},
                                                        {IE::Precision::I64, precision_e::I64},
                                                        {IE::Precision::U64, precision_e::U64},
                                                        {IE::Precision::BIN, precision_e::BIN},
                                                        {IE::Precision::CUSTOM, precision_e::CUSTOM}};

std::map<IE::Layout, layout_e> layout_map = {{IE::Layout::ANY, layout_e::ANY},
                                                {IE::Layout::NCHW, layout_e::NCHW},
                                                {IE::Layout::NHWC, layout_e::NHWC},
                                                {IE::Layout::NCDHW, layout_e::NCDHW},
                                                {IE::Layout::NDHWC, layout_e::NDHWC},
                                                {IE::Layout::OIHW, layout_e::OIHW},
                                                {IE::Layout::SCALAR, layout_e::SCALAR},
                                                {IE::Layout::C, layout_e::C},
                                                {IE::Layout::CHW, layout_e::CHW},
                                                {IE::Layout::HW, layout_e::HW},
                                                {IE::Layout::NC, layout_e::NC},
                                                {IE::Layout::CN, layout_e::CN},
                                                {IE::Layout::BLOCKED, layout_e::BLOCKED}};

std::map<IE::ResizeAlgorithm, resize_alg_e> resize_alg_map = {{IE::ResizeAlgorithm::NO_RESIZE, resize_alg_e::NO_RESIZE},
                                                                {IE::ResizeAlgorithm::RESIZE_AREA, resize_alg_e::RESIZE_AREA},
                                                                {IE::ResizeAlgorithm::RESIZE_BILINEAR, resize_alg_e::RESIZE_BILINEAR}};

std::map<IE::ColorFormat, colorformat_e> colorformat_map = {{IE::ColorFormat::RAW, colorformat_e::RAW},
                                                            {IE::ColorFormat::RGB, colorformat_e::RGB},
                                                            {IE::ColorFormat::BGR, colorformat_e::BGR},
                                                            {IE::ColorFormat::BGRX, colorformat_e::BGRX},
                                                            {IE::ColorFormat::RGBX, colorformat_e::RGBX},
                                                            {IE::ColorFormat::NV12, colorformat_e::NV12},
                                                            {IE::ColorFormat::I420, colorformat_e::I420}};

#define CATCH_IE_EXCEPTION(StatusCode, ExceptionType) catch (const IE::ExceptionType&) {return IEStatusCode::StatusCode;}

#define CATCH_IE_EXCEPTIONS                                         \
        CATCH_IE_EXCEPTION(GENERAL_ERROR, GeneralError)             \
        CATCH_IE_EXCEPTION(NOT_IMPLEMENTED, NotImplemented)         \
        CATCH_IE_EXCEPTION(NETWORK_NOT_LOADED, NetworkNotLoaded)    \
        CATCH_IE_EXCEPTION(PARAMETER_MISMATCH, ParameterMismatch)   \
        CATCH_IE_EXCEPTION(NOT_FOUND, NotFound)                     \
        CATCH_IE_EXCEPTION(OUT_OF_BOUNDS, OutOfBounds)              \
        CATCH_IE_EXCEPTION(UNEXPECTED, Unexpected)                  \
        CATCH_IE_EXCEPTION(REQUEST_BUSY, RequestBusy)               \
        CATCH_IE_EXCEPTION(RESULT_NOT_READY, ResultNotReady)        \
        CATCH_IE_EXCEPTION(NOT_ALLOCATED, NotAllocated)             \
        CATCH_IE_EXCEPTION(INFER_NOT_STARTED, InferNotStarted)      \
        CATCH_IE_EXCEPTION(NETWORK_NOT_READ, NetworkNotRead)        \
        CATCH_IE_EXCEPTION(INFER_CANCELLED, InferCancelled)         \
        catch (...) {return IEStatusCode::UNEXPECTED;}

/**
 *@brief convert the config type data to map type data.
 */
std::map<std::string, std::string> config2Map(const ie_config_t *config) {
    std::map<std::string, std::string> m;
    const ie_config_t *tmp = config;

    while (tmp && tmp->name && tmp->value) {
        m[tmp->name] = tmp->value;
        tmp = tmp->next;
    }
    return m;
}

std::map<std::string, IE::Parameter> config2ParamMap(const ie_config_t *config) {
    std::map<std::string, IE::Parameter> param_map;
    const ie_config_t *tmp = config;

    while (tmp) {
        IE::Parameter param = IE::Parameter(std::string(tmp->value));
        param_map[tmp->name] = param;
        tmp = tmp->next;
    }
    return param_map;
}

/**
 *@brief convert the parameter.
 */
void parameter2IEparam(const IE::Parameter param, ie_param_t *ie_param) {
    if (param.is<std::string>()) {
        std::unique_ptr<char> params_temp(new char[param.as<std::string>().length() + 1]);
        ie_param->params = params_temp.release();
        memcpy(ie_param->params, param.as<std::string>().c_str(), param.as<std::string>().length() + 1);
    } else if (param.is<std::vector<std::string>>()) {
        auto val = param.as<std::vector<std::string>>();
        if (val.size() > 0) {
            std::string tmp = val[0];
            for (size_t i = 1; i < val.size(); ++i) {
                tmp = tmp + ", " + val[i];
            }

            std::unique_ptr<char[]> params_temp(new char[tmp.length() + 1]);
            ie_param->params = params_temp.release();
            memcpy(ie_param->params, tmp.c_str(), tmp.length() + 1);
        } else {
            std::unique_ptr<char[]> params_temp(new char[1]);
            ie_param->params = params_temp.release();
            memcpy(ie_param->params, "", sizeof(char));
        }
    } else if (param.is<std::tuple<unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int >>();
        ie_param->range_for_streams[0] = std::get<0>(val);
        ie_param->range_for_streams[1] = std::get<1>(val);
    } else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int, unsigned int >>();
        ie_param->range_for_async_infer_request[0] = std::get<0>(val);
        ie_param->range_for_async_infer_request[1] = std::get<1>(val);
        ie_param->range_for_async_infer_request[2] = std::get<2>(val);
    } else if (param.is<unsigned int>()) {
        auto val = param.as<unsigned int>();
        ie_param->number = val;
    }
}

ie_version_t ie_c_api_version(void) {
    auto version = IE::GetInferenceEngineVersion();
    std::string version_str = version->buildNumber;

    ie_version_t version_res;
    std::unique_ptr<char[]> ver(new char[version_str.length() + 1]);
    version_res.api_version = ver.release();
    memcpy(version_res.api_version, version_str.c_str(), version_str.length() + 1);

    return version_res;
}

void ie_version_free(ie_version_t *version) {
    if (version) {
        delete[] version->api_version;
        version->api_version = NULL;
    }
}

void ie_param_free(ie_param_t *param) {
    if (param && param->params) {
        delete[] param->params;
        param->params = NULL;
    }
}

IEStatusCode ie_core_create(const char *xml_config_file, ie_core_t **core) {
    if (xml_config_file == nullptr || core == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IEStatusCode status = IEStatusCode::OK;
    try {
        std::unique_ptr<ie_core_t> tmp(new ie_core_t);
        tmp->object = IE::Core(xml_config_file);
        *core = tmp.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

void ie_core_free(ie_core_t **core) {
    if (core) {
        delete *core;
        *core = NULL;
    }
}

IEStatusCode ie_core_get_versions(const ie_core_t *core, const char *device_name, ie_core_versions_t *versions) {
    if (core == nullptr || device_name == nullptr || versions == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IEStatusCode status = IEStatusCode::OK;
    try {
        std::map<std::string, IE::Version> IEversions = core->object.GetVersions(device_name);
        size_t num = IEversions.size();
        if (num == 0) {
            return IEStatusCode::NOT_FOUND;
        }

        std::unique_ptr<ie_core_version_t[]> vers_ptrs(new ie_core_version_t[num]);
        assert(vers_ptrs);

        versions->num_vers = num;

        std::map<std::string, IE::Version>::iterator iter = IEversions.begin();
        for (size_t i = 0; i < num; ++i, ++iter) {
            std::unique_ptr<char[]> deviceName(new char[iter->first.length() + 1]);
            char *_deviceName = deviceName.release();
            memcpy(_deviceName, iter->first.c_str(), iter->first.length() + 1);
            vers_ptrs[i].device_name = _deviceName;
            IE_SUPPRESS_DEPRECATED_START
            vers_ptrs[i].major = iter->second.apiVersion.major;
            vers_ptrs[i].minor = iter->second.apiVersion.minor;
            IE_SUPPRESS_DEPRECATED_END
            vers_ptrs[i].build_number = iter->second.buildNumber;
            vers_ptrs[i].description = iter->second.description;
        }
        versions->versions = vers_ptrs.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

void ie_core_versions_free(ie_core_versions_t *vers) {
    if (vers) {
        for (size_t i = 0; i < vers->num_vers; ++i) {
            delete[] const_cast<char *>(vers->versions[i].device_name);
            vers->versions[i].device_name = NULL;
        }
        delete[] vers->versions;
        vers->versions = NULL;
    }
}

IEStatusCode ie_core_read_network(ie_core_t *core, const char *xml, const char *weights_file, ie_network_t **network) {
    if (core == nullptr || xml == nullptr || network == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IEStatusCode status = IEStatusCode::OK;

    try {
        std::unique_ptr<ie_network_t> network_result(new ie_network_t);
        std::string bin = "";
        if (weights_file) {
            bin = weights_file;
        }
        network_result->object = core->object.ReadNetwork(xml, bin);
        *network = network_result.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_read_network_from_memory(ie_core_t *core, const uint8_t *xml_content, size_t xml_content_size, \
        const ie_blob_t *weight_blob, ie_network_t **network) {
    if (core == nullptr || xml_content == nullptr || network == nullptr || weight_blob == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IEStatusCode status = IEStatusCode::OK;

    try {
        std::unique_ptr<ie_network_t> network_result(new ie_network_t);
        network_result->object = core->object.ReadNetwork(std::string(reinterpret_cast<const char *>(xml_content),
            reinterpret_cast<const char *>(xml_content + xml_content_size)), weight_blob->object);
        *network = network_result.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_load_network(ie_core_t *core, const ie_network_t *network, const char *device_name, \
        const ie_config_t *config, ie_executable_network_t **exe_network) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || network == nullptr || device_name == nullptr || config == nullptr || exe_network == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        std::map<std::string, std::string> conf_map;
        conf_map = config2Map(config);
        std::unique_ptr<ie_executable_network_t> exe_net(new ie_executable_network_t);

        // create plugin in the registery and then create ExecutableNetwork.
        exe_net->object = core->object.LoadNetwork(network->object, device_name, conf_map);
        *exe_network = exe_net.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_load_network_from_file(ie_core_t *core, const char *xml, const char *device_name, \
        const ie_config_t *config, ie_executable_network_t **exe_network) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || xml == nullptr || device_name == nullptr || config == nullptr || exe_network == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        std::map<std::string, std::string> conf_map;
        conf_map = config2Map(config);
        std::unique_ptr<ie_executable_network_t> exe_net(new ie_executable_network_t);

        exe_net->object = core->object.LoadNetwork(xml, device_name, conf_map);
        *exe_network = exe_net.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_set_config(ie_core_t *core, const ie_config_t *ie_core_config, const char *device_name) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || ie_core_config == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    const std::map<std::string, std::string> conf_map = config2Map(ie_core_config);
    std::string deviceName;
    if (device_name != nullptr) {
        deviceName = std::string(device_name);
    }

    try {
        core->object.SetConfig(conf_map, deviceName);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_register_plugin(ie_core_t *core, const char *plugin_name, const char *device_name ) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || plugin_name == nullptr || device_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        core->object.RegisterPlugin(plugin_name, device_name);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_register_plugins(ie_core_t *core, const char *xml_config_file) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || xml_config_file == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        core->object.RegisterPlugins(xml_config_file);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_unregister_plugin(ie_core_t *core, const char *device_name) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || device_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        core->object.UnregisterPlugin(device_name);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_add_extension(ie_core_t *core, const char *extension_path, const char *device_name) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || extension_path == nullptr || device_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        auto extension_ptr = std::make_shared<InferenceEngine::Extension>(std::string{extension_path});
        auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
        core->object.AddExtension(extension, device_name);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_get_metric(const ie_core_t *core, const char *device_name, const char *metric_name, ie_param_t *param_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || device_name == nullptr || metric_name == nullptr || param_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::Parameter param = core->object.GetMetric(device_name, metric_name);
        parameter2IEparam(param, param_result);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_get_config(const ie_core_t *core, const char *device_name, const char *config_name, ie_param_t *param_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (core == nullptr || device_name == nullptr || config_name == nullptr || param_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::Parameter param = core->object.GetConfig(device_name, config_name);

        // convert the parameter to ie_param_t
        parameter2IEparam(param, param_result);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_core_get_available_devices(const ie_core_t *core, ie_available_devices_t *avai_devices) {
    if (core == nullptr || avai_devices == nullptr)
        return IEStatusCode::GENERAL_ERROR;

    try {
        std::vector<std::string> _devices = core->object.GetAvailableDevices();
        avai_devices->num_devices = _devices.size();
        std::unique_ptr<char*[]> dev_ptrs(new char*[avai_devices->num_devices]);
        assert(dev_ptrs);

        for (size_t i = 0; i < avai_devices->num_devices; ++i) {
            std::unique_ptr<char[]> device_name(new char[_devices[i].length() + 1]);
            assert(device_name);
            dev_ptrs[i] = device_name.release();
            memcpy(dev_ptrs[i], _devices[i].c_str(), _devices[i].length() + 1);
        }
        avai_devices->devices = dev_ptrs.release();
    } CATCH_IE_EXCEPTIONS

    return IEStatusCode::OK;
}

void ie_core_available_devices_free(ie_available_devices_t *avai_devices) {
    if (avai_devices->devices) {
        for (size_t i = 0; i < avai_devices->num_devices; ++i) {
            if (avai_devices->devices[i]) {
                delete[] avai_devices->devices[i];
                avai_devices->devices[i] = NULL;
            }
        }
        delete[] avai_devices->devices;
        avai_devices->devices = NULL;
        avai_devices->num_devices = 0;
    }
}

void ie_exec_network_free(ie_executable_network_t **ie_exec_network) {
    if (ie_exec_network) {
        delete *ie_exec_network;
        *ie_exec_network = NULL;
    }
}

IEStatusCode ie_exec_network_create_infer_request(ie_executable_network_t *ie_exec_network, ie_infer_request_t **request) {
    IEStatusCode status = IEStatusCode::OK;
    if (ie_exec_network == nullptr || request == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return IEStatusCode::GENERAL_ERROR;
    }

    try {
        std::unique_ptr<ie_infer_request_t> req(new ie_infer_request_t);
        req->object = ie_exec_network->object.CreateInferRequest();
        *request = req.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_exec_network_get_metric(const ie_executable_network_t *ie_exec_network, const char *metric_name, ie_param_t *param_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (ie_exec_network == nullptr || metric_name == nullptr || param_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        InferenceEngine::Parameter parameter = ie_exec_network->object.GetMetric(metric_name);
        parameter2IEparam(parameter, param_result);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_exec_network_set_config(ie_executable_network_t *ie_exec_network, const ie_config_t *param_config) {
    IEStatusCode status = IEStatusCode::OK;

    if (ie_exec_network == nullptr || param_config == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        const std::map<std::string, IE::Parameter> conf_map = config2ParamMap(param_config);
        ie_exec_network->object.SetConfig(conf_map);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_exec_network_get_config(const ie_executable_network_t *ie_exec_network, const char *metric_config, ie_param_t *param_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (ie_exec_network == nullptr || metric_config == nullptr || param_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        InferenceEngine::Parameter parameter = ie_exec_network->object.GetConfig(metric_config);
        parameter2IEparam(parameter, param_result);
    } CATCH_IE_EXCEPTIONS

    return status;
}

void ie_network_free(ie_network_t **network) {
    if (network) {
        delete *network;
        *network = NULL;
    }
}

IEStatusCode ie_network_get_name(const ie_network_t *network, char **name) {
    if (network == nullptr || name == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    try {
        std::string _name = network->object.getName();
        std::unique_ptr<char[]> netName(new char[_name.length() + 1]);
        *name = netName.release();
        memcpy(*name, _name.c_str(), _name.length() + 1);
    } CATCH_IE_EXCEPTIONS

    return IEStatusCode::OK;
}

IEStatusCode ie_network_get_inputs_number(const ie_network_t *network, size_t *size_result) {
    IEStatusCode status = IEStatusCode::OK;
    if (network == nullptr || size_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        *size_result = inputs.size();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_input_name(const ie_network_t *network, size_t number, char **name) {
    if (network == nullptr || name == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IEStatusCode status = IEStatusCode::OK;

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();

        // check if the number is out of bounds.
        if (number < 0 || number >= inputs.size()) {
            status = IEStatusCode::OUT_OF_BOUNDS;
        } else {
            IE::InputsDataMap::iterator iter = inputs.begin();
            for (size_t i = 0; i < number; ++i) {
                ++iter;
            }
            std::unique_ptr<char[]> inputName(new char[iter->first.length() + 1]);
            *name = inputName.release();
            memcpy(*name, iter->first.c_str(), iter->first.length() + 1);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_input_precision(const ie_network_t *network, const char *input_name, precision_e *prec_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr || prec_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Precision p = inputs[input_name]->getPrecision();
            *prec_result = precision_map[p];
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_set_input_precision(ie_network_t *network, const char *input_name, const precision_e p) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Precision precision;
            for (auto it : precision_map) {
                if (it.second == p) {
                    precision = it.first;
                    break;
                }
            }
            inputs[input_name]->setPrecision(precision);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_input_layout(const ie_network_t *network, const char *input_name, layout_e *layout_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr || layout_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Layout l = inputs[input_name]->getLayout();
            *layout_result = layout_map[l];
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_set_input_layout(ie_network_t *network, const char *input_name, const layout_e l) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Layout layout = IE::Layout::NCHW;
            for (auto it : layout_map) {
                if (it.second == l) {
                    layout = it.first;
                    break;
                }
            }
            inputs[input_name]->setLayout(layout);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_input_dims(const ie_network_t *network, const char *input_name, dimensions_t *dims_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr || dims_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::SizeVector dims = inputs[input_name]->getTensorDesc().getDims();
            dims_result->ranks = dims.size();
            for (size_t i = 0; i< dims_result->ranks; ++i) {
                dims_result->dims[i] = dims[i];
            }
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_input_resize_algorithm(const ie_network_t *network, const char *input_name, resize_alg_e *resize_alg_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr || resize_alg_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::ResizeAlgorithm resize = inputs[input_name]->getPreProcess().getResizeAlgorithm();
            *resize_alg_result = resize_alg_map[resize];
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_set_input_resize_algorithm(ie_network_t *network, const char *input_name, const resize_alg_e resize_algo) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::ResizeAlgorithm resize = IE::ResizeAlgorithm::NO_RESIZE;
            for (auto it : resize_alg_map) {
                if (it.second == resize_algo) {
                    resize = it.first;
                    break;
                }
            }
            inputs[input_name]->getPreProcess().setResizeAlgorithm(resize);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_color_format(const ie_network_t *network, const char *input_name, colorformat_e *colformat_result) {
     IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr || colformat_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
            IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::ColorFormat color = inputs[input_name]->getPreProcess().getColorFormat();
            *colformat_result = colorformat_map[color];
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_set_color_format(ie_network_t *network, const char *input_name, const colorformat_e color_format) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || input_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::InputsDataMap inputs = network->object.getInputsInfo();
        if (inputs.find(input_name) == inputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::ColorFormat color = IE::ColorFormat::RGB;
            for (auto it : colorformat_map) {
                if (it.second == color_format) {
                    color = it.first;
                    break;
                }
            }
            inputs[input_name]->getPreProcess().setColorFormat(color);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_input_shapes(ie_network *network, input_shapes_t *shapes) {
    if (network == nullptr || shapes == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IEStatusCode status = IEStatusCode::OK;
    try {
        IE::ICNNNetwork::InputShapes net_shapes =  network->object.getInputShapes();
        size_t num = net_shapes.size();

        std::unique_ptr<input_shape[]> shape_ptrs(new input_shape[num]);

        assert(shape_ptrs);

        shapes->shape_num = num;

        IE::ICNNNetwork::InputShapes::iterator iter = net_shapes.begin();
        for (size_t i = 0; i < num; ++i, ++iter) {
            IE::SizeVector net_dim = iter->second;

            std::unique_ptr<char[]> _name(new char[iter->first.length() + 1]);
            shape_ptrs[i].name = _name.release();
            memcpy(shape_ptrs[i].name, iter->first.c_str(), iter->first.length() + 1);

            shape_ptrs[i].shape.ranks = net_dim.size();
            for (size_t j = 0; j < shape_ptrs[i].shape.ranks; ++j) {
                shape_ptrs[i].shape.dims[j] = net_dim[j];
            }
        }
        shapes->shapes = shape_ptrs.release();
        status = IEStatusCode::OK;
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_reshape(ie_network_t *network, const input_shapes_t shapes) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::ICNNNetwork::InputShapes net_shapes;
        for (size_t i = 0; i < shapes.shape_num; ++i) {
            IE::SizeVector net_dim;
            for (size_t j = 0; j < shapes.shapes[i].shape.ranks; ++j) {
                net_dim.push_back(shapes.shapes[i].shape.dims[j]);
            }

            net_shapes[shapes.shapes[i].name] = net_dim;
        }

        network->object.reshape(net_shapes);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_outputs_number(const ie_network_t *network, size_t *size_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || size_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::OutputsDataMap outputs = network->object.getOutputsInfo();
        *size_result = outputs.size();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_output_name(const ie_network_t *network, const size_t number, char **name) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::OutputsDataMap outputs = network->object.getOutputsInfo();
        // check if the number is out of bounds.
        if (number < 0 || number >= outputs.size()) {
            status = IEStatusCode::OUT_OF_BOUNDS;
        } else {
            IE::OutputsDataMap::iterator iter = outputs.begin();
            for (size_t i = 0; i < number; ++i) {
                ++iter;
            }
            std::unique_ptr<char[]> outputName(new char[iter->first.length() + 1]);
            *name = outputName.release();
            memcpy(*name, iter->first.c_str(), iter->first.length() + 1);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_output_precision(const ie_network_t *network, const char *output_name, precision_e *prec_result) {
     IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || output_name == nullptr || prec_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::OutputsDataMap outputs = network->object.getOutputsInfo();
        if (outputs.find(output_name) == outputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Precision p = outputs[output_name]->getPrecision();
            *prec_result = precision_map[p];
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_set_output_precision(ie_network_t *network, const char *output_name, const precision_e p) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || output_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::OutputsDataMap outputs = network->object.getOutputsInfo();
        if (outputs.find(output_name) == outputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Precision precision;
            for (auto it : precision_map) {
                if (it.second == p) {
                    precision = it.first;
                    break;
                }
            }
            outputs[output_name]->setPrecision(precision);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_output_layout(const ie_network_t *network, const char *output_name, layout_e *layout_result) {
     IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || output_name == nullptr || layout_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::OutputsDataMap outputs = network->object.getOutputsInfo();
        if (outputs.find(output_name) == outputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Layout l = outputs[output_name]->getLayout();
            *layout_result = layout_map[l];
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_set_output_layout(ie_network_t *network, const char *output_name, const layout_e l) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || output_name == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::OutputsDataMap outputs = network->object.getOutputsInfo();
        if (outputs.find(output_name) == outputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::Layout layout = IE::Layout::NCHW;
            for (auto it : layout_map) {
                if (it.second == l) {
                    layout = it.first;
                    break;
                }
            }
            outputs[output_name]->setLayout(layout);
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_network_get_output_dims(const ie_network_t *network, const char *output_name, dimensions_t *dims_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (network == nullptr || output_name == nullptr || dims_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::OutputsDataMap outputs = network->object.getOutputsInfo();
        if (outputs.find(output_name) == outputs.end()) {
            status = IEStatusCode::NOT_FOUND;
        } else {
            IE::SizeVector dims = outputs[output_name]->getTensorDesc().getDims();
            dims_result->ranks = dims.size();
            for (size_t i = 0; i< dims_result->ranks; ++i) {
                dims_result->dims[i] = dims[i];
            }
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

void ie_network_input_shapes_free(input_shapes_t *inputShapes) {
    if (inputShapes) {
        for (size_t i = 0; i < inputShapes->shape_num; ++i) {
            delete[] inputShapes->shapes[i].name;
            inputShapes->shapes[i].name = NULL;
        }
        delete[] inputShapes->shapes;
        inputShapes->shapes = NULL;
    }
}

void ie_network_name_free(char **name) {
    if (*name) {
        delete[] *name;
        *name = NULL;
    }
}

void ie_infer_request_free(ie_infer_request_t **infer_request) {
    if (infer_request) {
        delete *infer_request;
        *infer_request = NULL;
    }
}

IEStatusCode ie_infer_request_get_blob(ie_infer_request_t *infer_request, const char *name, ie_blob_t **blob) {
    IEStatusCode status = IEStatusCode::OK;

    if (infer_request == nullptr || name == nullptr || blob == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::Blob::Ptr blob_ptr = infer_request->object.GetBlob(name);
        std::unique_ptr<ie_blob_t> blob_result(new ie_blob_t);
        blob_result->object = blob_ptr;
        *blob = blob_result.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_infer_request_set_blob(ie_infer_request_t *infer_request, const char *name, const ie_blob_t *blob) {
    IEStatusCode status = IEStatusCode::OK;

    if (infer_request == nullptr || name ==nullptr || blob == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        infer_request->object.SetBlob(name, blob->object);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_infer_request_infer(ie_infer_request_t *infer_request) {
    IEStatusCode status = IEStatusCode::OK;

    if (infer_request == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        infer_request->object.Infer();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_infer_request_infer_async(ie_infer_request_t *infer_request) {
    IEStatusCode status = IEStatusCode::OK;

    if (infer_request == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        infer_request->object.StartAsync();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_infer_set_completion_callback(ie_infer_request_t *infer_request, ie_complete_call_back_t *callback) {
    IEStatusCode status = IEStatusCode::OK;

    if (infer_request == nullptr || callback == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        auto fun = [=]() {
            callback->completeCallBackFunc(callback->args);
        };
        infer_request->object.SetCompletionCallback(fun);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_infer_request_wait(ie_infer_request_t *infer_request, const int64_t timeout) {
    IEStatusCode status = IEStatusCode::OK;

    if (infer_request == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::StatusCode status_code = infer_request->object.Wait(timeout);
        status = status_map[status_code];
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_infer_request_set_batch(ie_infer_request_t *infer_request, const size_t size) {
    IEStatusCode status = IEStatusCode::OK;

    if (infer_request == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        infer_request->object.SetBatch(size);
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_blob_make_memory(const tensor_desc_t *tensorDesc, ie_blob_t **blob) {
    if (tensorDesc == nullptr || blob == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IE::Precision prec;
    for (auto it : precision_map) {
        if (it.second == tensorDesc->precision) {
            prec = it.first;
            break;
        }
    }

    IE::Layout l = IE::Layout::NCHW;
    for (auto it : layout_map) {
        if (it.second == tensorDesc->layout) {
            l = it.first;
            break;
        }
    }

    IE::SizeVector dims_vector;
    for (size_t i = 0; i < tensorDesc->dims.ranks; ++i) {
        dims_vector.push_back(tensorDesc->dims.dims[i]);
    }

    IEStatusCode status = IEStatusCode::OK;
    try {
        std::unique_ptr<ie_blob_t> _blob(new ie_blob_t);
        IE::TensorDesc tensor(prec, dims_vector, l);

        if (prec == IE::Precision::U8) {
            _blob->object = IE::make_shared_blob<uint8_t>(tensor);
        } else if (prec == IE::Precision::U16) {
            _blob->object = IE::make_shared_blob<uint16_t>(tensor);
        } else if (prec == IE::Precision::I8 || prec == IE::Precision::BIN || prec == IE::Precision::I4 || prec == IE::Precision::U4) {
            _blob->object = IE::make_shared_blob<int8_t>(tensor);
        } else if (prec == IE::Precision::I16 || prec == IE::Precision::FP16 || prec == IE::Precision::Q78) {
            _blob->object = IE::make_shared_blob<int16_t>(tensor);
        } else if (prec == IE::Precision::I32) {
            _blob->object = IE::make_shared_blob<int32_t>(tensor);
        } else if (prec == IE::Precision::U32) {
            _blob->object = IE::make_shared_blob<uint32_t>(tensor);
        } else if (prec == IE::Precision::I64) {
            _blob->object = IE::make_shared_blob<int64_t>(tensor);
        } else if (prec == IE::Precision::U64) {
            _blob->object = IE::make_shared_blob<uint64_t>(tensor);
        } else if  (prec == IE::Precision::FP32) {
            _blob->object = IE::make_shared_blob<float>(tensor);
        }  else if  (prec == IE::Precision::FP64) {
            _blob->object = IE::make_shared_blob<double>(tensor);
        } else {
            _blob->object = IE::make_shared_blob<uint8_t>(tensor);
        }

        _blob->object->allocate();
        *blob = _blob.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_blob_make_memory_from_preallocated(const tensor_desc_t *tensorDesc, void *ptr, size_t size, ie_blob_t **blob) {
    if (tensorDesc == nullptr || ptr == nullptr || blob == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IE::Precision prec;
    for (auto it : precision_map) {
        if (it.second == tensorDesc->precision) {
            prec = it.first;
            break;
        }
    }

    IE::Layout l = IE::Layout::NCHW;
    for (auto it : layout_map) {
        if (it.second == tensorDesc->layout) {
            l = it.first;
            break;
        }
    }

    IE::SizeVector dims_vector;
    for (size_t i = 0; i < tensorDesc->dims.ranks; ++i) {
        dims_vector.push_back(tensorDesc->dims.dims[i]);
    }

    IEStatusCode status = IEStatusCode::OK;
    try {
        IE::TensorDesc tensor(prec, dims_vector, l);
        std::unique_ptr<ie_blob_t> _blob(new ie_blob_t);
        if (prec == IE::Precision::U8) {
            uint8_t *p = reinterpret_cast<uint8_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if (prec == IE::Precision::U16) {
            uint16_t *p = reinterpret_cast<uint16_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if (prec == IE::Precision::I8 || prec == IE::Precision::BIN || prec == IE::Precision::I4 || prec == IE::Precision::U4) {
            int8_t *p = reinterpret_cast<int8_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if (prec == IE::Precision::I16 || prec == IE::Precision::FP16 || prec == IE::Precision::Q78) {
            int16_t *p = reinterpret_cast<int16_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if (prec == IE::Precision::I32) {
            int32_t *p = reinterpret_cast<int32_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if (prec == IE::Precision::U32) {
            uint32_t *p = reinterpret_cast<uint32_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if (prec == IE::Precision::I64) {
            int64_t *p = reinterpret_cast<int64_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if (prec == IE::Precision::U64) {
            uint64_t *p = reinterpret_cast<uint64_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if  (prec == IE::Precision::FP32) {
            float *p = reinterpret_cast<float *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else if  (prec == IE::Precision::FP64) {
            double *p = reinterpret_cast<double *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        } else {
            uint8_t *p = reinterpret_cast<uint8_t *>(ptr);
            _blob->object = IE::make_shared_blob(tensor, p, size);
        }
        *blob = _blob.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_blob_make_memory_with_roi(const ie_blob_t *inputBlob, const roi_t *roi, ie_blob_t **blob) {
    if (inputBlob == nullptr || roi == nullptr || blob == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    IEStatusCode status = IEStatusCode::OK;
    try {
        std::unique_ptr<ie_blob_t> _blob(new ie_blob_t);
        IE::ROI roi_d = {roi->id, roi->posX, roi->posY, roi->sizeX, roi->sizeY};
        _blob->object = IE::make_shared_blob(inputBlob->object, roi_d);
        *blob = _blob.release();
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_blob_make_memory_nv12(const ie_blob_t *y, const ie_blob_t *uv, ie_blob_t **nv12Blob) {
    if (y == nullptr || uv == nullptr || nv12Blob == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    try {
        std::unique_ptr<ie_blob_t> _blob(new ie_blob_t);
        _blob->object = IE::make_shared_blob<IE::NV12Blob>(y->object, uv->object);
        *nv12Blob = _blob.release();
    } CATCH_IE_EXCEPTIONS

    return IEStatusCode::OK;
}

IEStatusCode ie_blob_make_memory_i420(const ie_blob_t *y, const ie_blob_t *u, const ie_blob_t *v, ie_blob_t **i420Blob) {
    if (y == nullptr || u == nullptr || v == nullptr || i420Blob == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    try {
        std::unique_ptr<ie_blob_t> _blob(new ie_blob_t);
        _blob->object = IE::make_shared_blob<IE::I420Blob>(y->object, u->object, v->object);
        *i420Blob = _blob.release();
    } CATCH_IE_EXCEPTIONS

    return IEStatusCode::OK;
}

IEStatusCode ie_blob_size(ie_blob_t *blob, int *size_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (blob == nullptr || size_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    *size_result = blob->object->size();

    return status;
}

IEStatusCode ie_blob_byte_size(ie_blob_t *blob, int *bsize_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (blob == nullptr || bsize_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    *bsize_result = blob->object->byteSize();

    return status;
}

void ie_blob_deallocate(ie_blob_t **blob) {
    if (*blob) {
        (*blob)->object->deallocate();
        delete *blob;
        *blob = NULL;
    }
}

IEStatusCode ie_blob_get_buffer(const ie_blob_t *blob, ie_blob_buffer_t *blob_buffer) {
    if (blob == nullptr || blob_buffer == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    blob_buffer->buffer =  blob->object->buffer();

    return IEStatusCode::OK;
}

IEStatusCode ie_blob_get_cbuffer(const ie_blob_t *blob, ie_blob_buffer_t *blob_cbuffer) {
    if (blob == nullptr || blob_cbuffer == nullptr) {
        return IEStatusCode::GENERAL_ERROR;
    }

    blob_cbuffer->cbuffer =  blob->object->cbuffer();

    return IEStatusCode::OK;
}

IEStatusCode ie_blob_get_dims(const ie_blob_t *blob, dimensions_t *dims_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (blob == nullptr || dims_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::SizeVector size_vector = blob->object->getTensorDesc().getDims();
        dims_result->ranks = size_vector.size();
        for (size_t i = 0; i< dims_result->ranks; ++i) {
            dims_result->dims[i] = size_vector[i];
        }
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_blob_get_layout(const ie_blob_t *blob, layout_e *layout_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (blob == nullptr || layout_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::Layout l = blob->object->getTensorDesc().getLayout();
        *layout_result = layout_map[l];
    } CATCH_IE_EXCEPTIONS

    return status;
}

IEStatusCode ie_blob_get_precision(const ie_blob_t *blob, precision_e *prec_result) {
    IEStatusCode status = IEStatusCode::OK;

    if (blob == nullptr || prec_result == nullptr) {
        status = IEStatusCode::GENERAL_ERROR;
        return status;
    }

    try {
        IE::Precision p = blob->object->getTensorDesc().getPrecision();
        *prec_result = precision_map[p];
    } CATCH_IE_EXCEPTIONS

    return status;
}

void ie_blob_free(ie_blob_t **blob) {
    if (blob) {
        delete *blob;
        *blob = NULL;
    }
}
