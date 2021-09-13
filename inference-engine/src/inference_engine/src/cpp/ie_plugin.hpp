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
#include "so_ptr.hpp"
#include "openvino/runtime/common.hpp"

#define PLUGIN_CALL_STATEMENT(...)                                                                \
    if (!_impl) IE_THROW() << "Wrapper used in the PLUGIN_CALL_STATEMENT was not initialized.";   \
    try {                                                                                         \
        __VA_ARGS__;                                                                              \
    } catch(...) {::InferenceEngine::details::Rethrow();}
namespace ov {
namespace runtime {

/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
struct InferencePlugin {
    std::shared_ptr<void> _so;
    std::shared_ptr<ie::IInferencePlugin> _impl;

    InferencePlugin(const std::shared_ptr<void>& so, const std::shared_ptr<ie::IInferencePlugin>& impl) :
        _so{so},
        _impl{impl} {
        if (_impl == nullptr) IE_THROW() << "InferencePlugin was not initialized";
    }

    void set_name(const std::string& deviceName) {
        PLUGIN_CALL_STATEMENT(_impl->SetName(deviceName));
    }

    void set_core(std::weak_ptr<ie::ICore> core) {
        PLUGIN_CALL_STATEMENT(_impl->SetCore(core));
    }

    const ie::Version get_version() const {
        PLUGIN_CALL_STATEMENT(return _impl->GetVersion());
    }

    void add_extension(const ie::IExtensionPtr& extension) {
        PLUGIN_CALL_STATEMENT(_impl->AddExtension(extension));
    }

    void set_config(const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(_impl->SetConfig(config));
    }

    SoPtr<ie::IExecutableNetworkInternal> load_model(const ie::CNNNetwork& network, const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->LoadNetwork(network, config)});
    }

    SoPtr<ie::IExecutableNetworkInternal> load_model(const ie::CNNNetwork& network,
                                                               const std::shared_ptr<ie::IRemoteContext>& context,
                                                               const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->LoadNetwork(network, config, context)});
    }

    SoPtr<ie::IExecutableNetworkInternal> load_model(const std::string& modelPath, const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->LoadNetwork(modelPath, config)});
    }

    ie::QueryNetworkResult query_model(const ie::CNNNetwork& network,
                                       const ConfigMap& config) const {
        ie::QueryNetworkResult res;
        PLUGIN_CALL_STATEMENT(res = _impl->QueryNetwork(network, config));
        if (res.rc != ie::OK) IE_THROW() << res.resp.msg;
        return res;
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(const std::string& modelFileName,
                                                                 const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->ImportNetwork(modelFileName, config)});
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(std::istream& networkModel,
                                    const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->ImportNetwork(networkModel, config)});
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(std::istream& networkModel,
                                                                 const std::shared_ptr<ie::IRemoteContext>& context,
                                                                 const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->ImportNetwork(networkModel, context, config)});
    }

    ie::Parameter get_metric(const std::string& name, const ie::ParamMap& options) const {
        PLUGIN_CALL_STATEMENT(return _impl->GetMetric(name, options));
    }

    SoPtr<ie::IRemoteContext> create_context(const ie::ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->CreateContext(params)});
    }

    SoPtr<ie::IRemoteContext> get_default_context(const ie::ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_so, _impl->GetDefaultContext(params)});
    }

    ie::Parameter get_config(const std::string& name, const ie::ParamMap& options) const {
        PLUGIN_CALL_STATEMENT(return _impl->GetConfig(name, options));
    }
};

}  // namespace runtime
}  // namespace ov

#undef PLUGIN_CALL_STATEMENT