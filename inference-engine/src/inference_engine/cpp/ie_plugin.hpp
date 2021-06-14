// Copyright (C) 2018-2021 Intel Corporation
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

#include "file_utils.h"
#include "cpp/ie_cnn_network.h"
#include "cpp/exception2status.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

#if defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wreturn-type"
#endif

#define PLUGIN_CALL_STATEMENT(...)                                                                \
    if (!_ptr) IE_THROW() << "Wrapper used in the PLUGIN_CALL_STATEMENT was not initialized.";    \
    try {                                                                                         \
        __VA_ARGS__;                                                                              \
    } catch(...) {details::Rethrow();}

namespace InferenceEngine {
/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class InferencePlugin : protected details::SOPointer<IInferencePlugin> {
    using details::SOPointer<IInferencePlugin>::SOPointer;
    friend class ICore;

public:
    void SetName(const std::string & deviceName) {
        PLUGIN_CALL_STATEMENT(_ptr->SetName(deviceName));
    }

    void SetCore(ICore* core) {
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

    details::SOPointer<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->LoadNetwork(network, config)});
    }

    details::SOPointer<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network,
                                                               const std::shared_ptr<RemoteContext>& context,
                                                               const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->LoadNetwork(network, config, context)});
    }

    details::SOPointer<IExecutableNetworkInternal> LoadNetwork(const std::string& modelPath, const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->LoadNetwork(modelPath, config)});
    }

    QueryNetworkResult QueryNetwork(const CNNNetwork& network,
                                    const std::map<std::string, std::string>& config) const {
        QueryNetworkResult res;
        PLUGIN_CALL_STATEMENT(res = _ptr->QueryNetwork(network, config));
        if (res.rc != OK) IE_THROW() << res.resp.msg;
        return res;
    }

    details::SOPointer<IExecutableNetworkInternal> ImportNetwork(const std::string& modelFileName,
                                                                 const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->ImportNetwork(modelFileName, config)});
    }

    details::SOPointer<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                    const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->ImportNetwork(networkModel, config)});
    }

    details::SOPointer<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                                                 const std::shared_ptr<RemoteContext>& context,
                                                                 const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->ImportNetwork(networkModel, context, config)});
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
        PLUGIN_CALL_STATEMENT(return _ptr->GetMetric(name, options));
    }

    std::shared_ptr<RemoteContext> CreateContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return _ptr->CreateContext(params));
    }

    std::shared_ptr<RemoteContext> GetDefaultContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return _ptr->GetDefaultContext(params));
    }

    Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
        PLUGIN_CALL_STATEMENT(return _ptr->GetConfig(name, options));
    }
};
}  // namespace InferenceEngine

#undef PLUGIN_CALL_STATEMENT

#if defined __GNUC__
# pragma GCC diagnostic pop
#endif
