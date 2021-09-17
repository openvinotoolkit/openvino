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

#if defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wreturn-type"
#endif

#define PLUGIN_CALL_STATEMENT(...)                                                                \
    if (!_ptr) IE_THROW() << "Wrapper used in the PLUGIN_CALL_STATEMENT was not initialized.";    \
    try {                                                                                         \
        __VA_ARGS__;                                                                              \
    } catch(...) {::InferenceEngine::details::Rethrow();}

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

    details::SOPointer<RemoteContext> CreateContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->CreateContext(params)});
    }

    details::SOPointer<RemoteContext> GetDefaultContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->GetDefaultContext(params)});
    }

    Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
        PLUGIN_CALL_STATEMENT(return _ptr->GetConfig(name, options));
    }
};
}  // namespace InferenceEngine


#if defined __GNUC__
# pragma GCC diagnostic pop
#endif

namespace ov {
namespace runtime {

/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
struct InferencePlugin {
    std::shared_ptr<void> _so;
    std::shared_ptr<ie::IInferencePlugin> _ptr;

    InferencePlugin(const std::shared_ptr<void>& so, const std::shared_ptr<ie::IInferencePlugin>& impl) :
        _so{so},
        _ptr{impl} {
        IE_ASSERT(_ptr != nullptr);
    }

    void set_name(const std::string& deviceName) {
        PLUGIN_CALL_STATEMENT(_ptr->SetName(deviceName));
    }

    void set_core(std::weak_ptr<ie::ICore> core) {
        PLUGIN_CALL_STATEMENT(_ptr->SetCore(core));
    }

    const ie::Version get_version() const {
        PLUGIN_CALL_STATEMENT(return _ptr->GetVersion());
    }

    void add_extension(const ie::IExtensionPtr& extension) {
        PLUGIN_CALL_STATEMENT(_ptr->AddExtension(extension));
    }

    void set_config(const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(_ptr->SetConfig(config));
    }

    SoPtr<ie::IExecutableNetworkInternal> load_model(const ie::CNNNetwork& network, const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->LoadNetwork(network, config)});
    }

    SoPtr<ie::IExecutableNetworkInternal> load_model(const ie::CNNNetwork& network,
                                                               const std::shared_ptr<ie::RemoteContext>& context,
                                                               const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->LoadNetwork(network, config, context)});
    }

    SoPtr<ie::IExecutableNetworkInternal> load_model(const std::string& modelPath, const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->LoadNetwork(modelPath, config)});
    }

    ie::QueryNetworkResult query_model(const ie::CNNNetwork& network,
                                       const ConfigMap& config) const {
        ie::QueryNetworkResult res;
        PLUGIN_CALL_STATEMENT(res = _ptr->QueryNetwork(network, config));
        if (res.rc != ie::OK) IE_THROW() << res.resp.msg;
        return res;
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(const std::string& modelFileName,
                                                                 const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->ImportNetwork(modelFileName, config)});
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(std::istream& networkModel,
                                    const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->ImportNetwork(networkModel, config)});
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(std::istream& networkModel,
                                                                 const std::shared_ptr<ie::RemoteContext>& context,
                                                                 const ConfigMap& config) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->ImportNetwork(networkModel, context, config)});
    }

    ie::Parameter get_metric(const std::string& name, const ie::ParamMap& options) const {
        PLUGIN_CALL_STATEMENT(return _ptr->GetMetric(name, options));
    }

    SoPtr<ie::RemoteContext> create_context(const ie::ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->CreateContext(params)});
    }

    SoPtr<ie::RemoteContext> get_default_context(const ie::ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_so, _ptr->GetDefaultContext(params)});
    }

    ie::Parameter get_config(const std::string& name, const ie::ParamMap& options) const {
        PLUGIN_CALL_STATEMENT(return _ptr->GetConfig(name, options));
    }
};

}  // namespace runtime
}  // namespace ov

#undef PLUGIN_CALL_STATEMENT