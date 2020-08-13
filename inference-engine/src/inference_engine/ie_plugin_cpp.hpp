// Copyright (C) 2018-2020 Intel Corporation
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
#include "cpp/ie_executable_network.hpp"
#include "cpp/ie_cnn_network.h"
#include "details/ie_exception_conversion.hpp"
#include "ie_plugin_ptr.hpp"

#if defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wreturn-type"
#endif

#define CALL_STATEMENT(...)                                                                        \
    if (!actual) THROW_IE_EXCEPTION << "Wrapper used in the CALL_STATEMENT was not initialized.";  \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } catch (const InferenceEngine::details::InferenceEngineException& iex) {                      \
        InferenceEngine::details::extract_exception(iex.hasStatus() ?                              \
            iex.getStatus() : GENERAL_ERROR, iex.what());                                          \
    } catch (const std::exception& ex) {                                                           \
        InferenceEngine::details::extract_exception(GENERAL_ERROR, ex.what());                     \
    } catch (...) {                                                                                \
        InferenceEngine::details::extract_exception(UNEXPECTED, "");                               \
    }

namespace InferenceEngine {

/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class InferencePlugin {
    InferenceEnginePluginPtr actual;

public:
    InferencePlugin() = default;

    explicit InferencePlugin(const InferenceEnginePluginPtr& pointer): actual(pointer) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "InferencePlugin wrapper was not initialized.";
        }
    }

    explicit InferencePlugin(const FileUtils::FilePath & libraryLocation) :
        actual(libraryLocation) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "InferencePlugin wrapper was not initialized.";
        }
    }

    void SetName(const std::string & deviceName) {
        CALL_STATEMENT(actual->SetName(deviceName));
    }

    void SetCore(ICore* core) {
        CALL_STATEMENT(actual->SetCore(core));
    }

    const Version GetVersion() const {
        CALL_STATEMENT(return actual->GetVersion());
    }

    ExecutableNetwork LoadNetwork(CNNNetwork network, const std::map<std::string, std::string>& config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATEMENT(actual->LoadNetwork(ret, network, config));
        return ExecutableNetwork(ret, actual);
    }

    void AddExtension(InferenceEngine::IExtensionPtr extension) {
        CALL_STATEMENT(actual->AddExtension(extension));
    }

    void SetConfig(const std::map<std::string, std::string>& config) {
        CALL_STATEMENT(actual->SetConfig(config));
    }

    ExecutableNetwork ImportNetwork(const std::string& modelFileName,
                                    const std::map<std::string, std::string>& config) {
        CALL_STATEMENT(return ExecutableNetwork(actual->ImportNetwork(modelFileName, config), actual));
    }

    void QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                      QueryNetworkResult& res) const {
        CALL_STATEMENT(actual->QueryNetwork(network, config, res));
        if (res.rc != OK) THROW_IE_EXCEPTION << res.resp.msg;
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const std::map<std::string, std::string> &config) {
        CALL_STATEMENT(return actual->ImportNetwork(networkModel, config));
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
        CALL_STATEMENT(return actual->GetMetric(name, options));
    }

    ExecutableNetwork LoadNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                  RemoteContext::Ptr context) {
        CALL_STATEMENT(return actual->LoadNetwork(network, config, context));
    }

    RemoteContext::Ptr CreateContext(const ParamMap& params) {
        CALL_STATEMENT(return actual->CreateContext(params));
    }

    RemoteContext::Ptr GetDefaultContext() {
        CALL_STATEMENT(return actual->GetDefaultContext());
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const RemoteContext::Ptr& context,
                                    const std::map<std::string, std::string>& config) {
        CALL_STATEMENT(return actual->ImportNetwork(networkModel, context, config));
    }

    Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
        CALL_STATEMENT(return actual->GetConfig(name, options));
    }

    /**
     * @brief Converts InferenceEngine to InferenceEnginePluginPtr pointer
     *
     * @return Wrapped object
     */
    operator InferenceEngine::InferenceEnginePluginPtr() {
        return actual;
    }

    /**
     * @brief Shared pointer on InferencePlugin object
     *
     */
    using Ptr = std::shared_ptr<InferencePlugin>;
};
}  // namespace InferenceEngine

#undef CALL_STATEMENT

#if defined __GNUC__
# pragma GCC diagnostic pop
#endif
