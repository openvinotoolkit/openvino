// Copyright (C) 2018-2022 Intel Corporation
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
struct InferencePlugin {
    std::shared_ptr<InferenceEngine::IInferencePlugin> _ptr;
    std::shared_ptr<void> _so;

    ~InferencePlugin() {
        _ptr = {};
    }

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

    ov::runtime::SoPtr<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(network, config), _so});
    }

    ov::runtime::SoPtr<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network,
                                                               const std::shared_ptr<RemoteContext>& context,
                                                               const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(network, config, context), _so});
    }

    ov::runtime::SoPtr<IExecutableNetworkInternal> LoadNetwork(const std::string& modelPath, const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(modelPath, config), _so});
    }

    QueryNetworkResult QueryNetwork(const CNNNetwork& network,
                                    const std::map<std::string, std::string>& config) const {
        QueryNetworkResult res;
        PLUGIN_CALL_STATEMENT(res = _ptr->QueryNetwork(network, config));
        if (res.rc != OK) IE_THROW() << res.resp.msg;
        return res;
    }

    ov::runtime::SoPtr<IExecutableNetworkInternal> ImportNetwork(const std::string& modelFileName,
                                                                 const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(modelFileName, config), _so});
    }

    ov::runtime::SoPtr<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                    const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(networkModel, config), _so});
    }

    ov::runtime::SoPtr<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                                                 const std::shared_ptr<RemoteContext>& context,
                                                                 const std::map<std::string, std::string>& config) {
        PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(networkModel, context, config), _so});
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
        PLUGIN_CALL_STATEMENT(return {_ptr->GetMetric(name, options), _so});
    }

    ov::runtime::SoPtr<RemoteContext> CreateContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_ptr->CreateContext(params), _so});
    }

    ov::runtime::SoPtr<RemoteContext> GetDefaultContext(const ParamMap& params) {
        PLUGIN_CALL_STATEMENT(return {_ptr->GetDefaultContext(params), _so});
    }

    Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
        PLUGIN_CALL_STATEMENT(return {_ptr->GetConfig(name, options), _so});
    }
};
}  // namespace InferenceEngine


#if defined __GNUC__
# pragma GCC diagnostic pop
#endif

namespace ov {
namespace runtime {

#define OV_PLUGIN_CALL_STATEMENT(...)                                         \
    OPENVINO_ASSERT(_ptr != nullptr, "InferencePlugin was not initialized."); \
    try {                                                                     \
        __VA_ARGS__;                                                          \
    } catch (...) {                                                           \
        ::InferenceEngine::details::Rethrow();                                \
    }

/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class InferencePlugin {
    std::shared_ptr<ie::IInferencePlugin> _ptr;
    std::shared_ptr<void> _so;

public:
    InferencePlugin() = default;

    ~InferencePlugin() {
        _ptr = {};
    }

    InferencePlugin(const std::shared_ptr<ie::IInferencePlugin>& ptr, const std::shared_ptr<void>& so) :
        _ptr{ptr},
        _so{so} {
        OPENVINO_ASSERT(_ptr != nullptr, "InferencePlugin was not initialized.");
    }

    void set_name(const std::string& deviceName) {
        OV_PLUGIN_CALL_STATEMENT(_ptr->SetName(deviceName));
    }

    void set_core(std::weak_ptr<ie::ICore> core) {
        OV_PLUGIN_CALL_STATEMENT(_ptr->SetCore(core));
    }

    const ie::Version get_version() const {
        OV_PLUGIN_CALL_STATEMENT(return _ptr->GetVersion());
    }

    void add_extension(const ie::IExtensionPtr& extension) {
        OV_PLUGIN_CALL_STATEMENT(_ptr->AddExtension(extension));
    }

    void set_config(const ConfigMap& config) {
        OV_PLUGIN_CALL_STATEMENT(_ptr->SetConfig(config));
    }

    SoPtr<ie::IExecutableNetworkInternal> compile_model(const ie::CNNNetwork& network, const ConfigMap& config) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(network, config), _so});
    }

    SoPtr<ie::IExecutableNetworkInternal> compile_model(const ie::CNNNetwork& network,
                                                        const std::shared_ptr<ie::RemoteContext>& context,
                                                        const ConfigMap& config) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(network, config, context), _so});
    }

    SoPtr<ie::IExecutableNetworkInternal> compile_model(const std::string& modelPath, const ConfigMap& config) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->LoadNetwork(modelPath, config), _so});
    }

    ie::QueryNetworkResult query_model(const ie::CNNNetwork& network,
                                       const ConfigMap& config) const {
        ie::QueryNetworkResult res;
        OV_PLUGIN_CALL_STATEMENT(res = _ptr->QueryNetwork(network, config));
        OPENVINO_ASSERT(res.rc == ie::OK, res.resp.msg);
        return res;
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(const std::string& modelFileName,
                                                       const ConfigMap& config) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(modelFileName, config), _so});
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(std::istream& networkModel,
                                    const ConfigMap& config) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(networkModel, config), _so});
    }

    SoPtr<ie::IExecutableNetworkInternal> import_model(std::istream& networkModel,
                                                       const std::shared_ptr<ie::RemoteContext>& context,
                                                       const ConfigMap& config) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->ImportNetwork(networkModel, context, config), _so});
    }

    ie::Parameter get_metric(const std::string& name, const ie::ParamMap& options) const {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->GetMetric(name, options), _so});
    }

    SoPtr<ie::RemoteContext> create_context(const ie::ParamMap& params) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->CreateContext(params), _so});
    }

    SoPtr<ie::RemoteContext> get_default_context(const ie::ParamMap& params) {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->GetDefaultContext(params), _so});
    }

    ie::Parameter get_config(const std::string& name, const ie::ParamMap& options) const {
        OV_PLUGIN_CALL_STATEMENT(return {_ptr->GetConfig(name, options), _so});
    }
};

}  // namespace runtime
}  // namespace ov

#undef PLUGIN_CALL_STATEMENT
#undef OV_PLUGIN_CALL_STATEMENT
