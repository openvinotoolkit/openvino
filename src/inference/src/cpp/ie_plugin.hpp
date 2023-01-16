// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine plugin C++ API
 *
 * @file ie_plugin_cpp.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "any_copy.hpp"
#include "cpp/exception2status.hpp"
#include "cpp/ie_cnn_network.h"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "file_utils.h"
#include "ie_plugin_config.hpp"
#include "openvino/runtime/common.hpp"
#include "so_ptr.hpp"

#if defined __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wreturn-type"
#endif

#define PLUGIN_CALL_STATEMENT(...)                                                      \
    if (!_ptr)                                                                          \
        IE_THROW() << "Wrapper used in the PLUGIN_CALL_STATEMENT was not initialized."; \
    try {                                                                               \
        __VA_ARGS__;                                                                    \
    } catch (...) {                                                                     \
        ::InferenceEngine::details::Rethrow();                                          \
    }

namespace InferenceEngine {
/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
struct InferencePlugin {
    std::shared_ptr<InferenceEngine::IInferencePlugin> _ptr;
    std::shared_ptr<void> _so;

    ~InferencePlugin() {
        _ptr = {};
    }

    void SetName(const std::string& deviceName) {
        PLUGIN_CALL_STATEMENT(_ptr->SetName(deviceName));
    }

    void SetCore(std::weak_ptr<InferenceEngine::ICore> core) {
        PLUGIN_CALL_STATEMENT(_ptr->SetCore(core));
    }

    const Version GetVersion() const {
        PLUGIN_CALL_STATEMENT(return _ptr->GetVersion());
    }

    void AddExtension(const InferenceEngine::IExtensionPtr& extension) {
        PLUGIN_CALL_STATEMENT(_ptr->AddExtension(extension));
    }

    void SetConfig(const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(_ptr->SetConfig(config));
    }

    ov::SoPtr<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network,
                                                      const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(network, config), _so});
    }

    ov::SoPtr<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network,
                                                      const std::shared_ptr<RemoteContext>& context,
                                                      const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(network, config, context), _so});
    }

    ov::SoPtr<IExecutableNetworkInternal> LoadNetwork(const std::string& modelPath,
                                                      const std::map<std::string, std::string>& config) {
        ov::SoPtr<IExecutableNetworkInternal> res;
        PLUGIN_CALL_STATEMENT(res = _ptr->LoadNetwork(modelPath, config));
        if (!res._so)
            res._so = _so;
        return res;
    }

    QueryNetworkResult QueryNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) const {
        QueryNetworkResult res;
        PLUGIN_CALL_STATEMENT(res = _ptr->QueryNetwork(network, config));
        if (res.rc != OK)
            IE_THROW() << res.resp.msg;
        return res;
    }

    ov::SoPtr<IExecutableNetworkInternal> ImportNetwork(const std::string& modelFileName,
                                                        const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(modelFileName, config), _so});
    }

    ov::SoPtr<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                                        const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(networkModel, config), _so});
    }

    ov::SoPtr<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                                        const std::shared_ptr<RemoteContext>& context,
                                                        const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(networkModel, context, config), _so});
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
        PLUGIN_CALL_STATEMENT(return {_ptr->GetMetric(name, options), {_so}});
    }

    ov::SoPtr<RemoteContext> CreateContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_ptr->CreateContext(params), {_so}});
    }

    ov::SoPtr<RemoteContext> GetDefaultContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_ptr->GetDefaultContext(params), {_so}});
    }

    Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
        PLUGIN_CALL_STATEMENT(return {_ptr->GetConfig(name, options), {_so}});
    }
};
}  // namespace InferenceEngine

#if defined __GNUC__
#    pragma GCC diagnostic pop
#endif

#undef PLUGIN_CALL_STATEMENT
