// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Inference Engine plugin API wrapper, to be used by particular implementors
 * @file ie_plugin_internal.hpp
 */

#pragma once

#include <ie_plugin_config.hpp>

#include <map>
#include <memory>
#include <string>
#include <limits>

#include "cpp_interfaces/base/ie_executable_network_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace InferenceEngine {

namespace {

/**
 * @private
 */
static inline void parsePluginName(std::istream& networkModel) {
    ExportMagic magic = {};
    auto currentPos = networkModel.tellg();
    networkModel.read(magic.data(), magic.size());
    auto exportedWithName = (exportMagic == magic);
    if (exportedWithName) {
        networkModel.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else {
        networkModel.seekg(currentPos, networkModel.beg);
    }
}

}  // namespace

/**
 * @brief Optimal implementation of IInferencePlugin interface to avoid duplication in all plugins
 * @ingroup ie_dev_api_plugin_api
 */
class InferencePluginInternal : public IInferencePlugin {
protected:
    /**
     * @brief Destroys the object.
     */
    ~InferencePluginInternal() override = default;

public:
    ExecutableNetwork LoadNetwork(const ICNNNetwork& network,
                                  const std::map<std::string, std::string>& config) override {
        return LoadNetworkImplPrivate(network, config);
    }

    ExecutableNetwork LoadNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                  RemoteContext::Ptr context) override {
        return LoadNetworkImplPrivate(network, config, context);;
    }

    IExecutableNetwork::Ptr ImportNetwork(const std::string& modelFileName,
                                          const std::map<std::string, std::string>& config) override {
        (void)modelFileName;
        (void)config;
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const std::map<std::string, std::string>& config) override {
        parsePluginName(networkModel);
        return ImportNetworkImpl(networkModel, config);
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const RemoteContext::Ptr& context,
                                    const std::map<std::string, std::string>& config) override {
        parsePluginName(networkModel);
        return ImportNetworkImpl(networkModel, context, config);
    }

    void SetConfig(const std::map<std::string, std::string>& config) override {
        (void)config;
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void SetCore(ICore* core) noexcept override {
        assert(nullptr != core);
        _core = core;
    }

    ICore* GetCore() const noexcept override {
        return _core;
    }

    void AddExtension(InferenceEngine::IExtensionPtr /*extension*/) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void QueryNetwork(const ICNNNetwork& /*network*/, const std::map<std::string, std::string>& /*config*/,
                      QueryNetworkResult& /*res*/) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void SetName(const std::string& pluginName) noexcept override {
        _pluginName = pluginName;
    }

    std::string GetName() const noexcept override {
        return _pluginName;
    }

    Parameter GetConfig(const std::string& /*name*/,
                        const std::map<std::string, Parameter>& /*options*/) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    Parameter GetMetric(const std::string& /*name*/,
                        const std::map<std::string, Parameter>& /*options*/) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    RemoteContext::Ptr CreateContext(const ParamMap& /*params*/) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    RemoteContext::Ptr GetDefaultContext() override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

private:
    /**
     * @brief A helper method which clones a ICNNNetwork object, keeps InputsDataMap and OutputsDataMap data maps,
     * and creates an IExecutableNetwork object
     * @param network An input ICNNNetwork object used to create an executable network object
     * @param config A map of string -> string configuration options.
     * @param context An optional pointer to RemoteContext
     * @return An output executable network object
     */
    ExecutableNetwork LoadNetworkImplPrivate(const ICNNNetwork& network,
                                             const std::map<std::string, std::string>& config,
                                             RemoteContext::Ptr context = nullptr) {
        InputsDataMap networkInputs, networkInputsCloned;
        OutputsDataMap networkOutputs, networkOutputsCloned;
        network.getInputsInfo(networkInputs);
        network.getOutputsInfo(networkOutputs);
        copyInputOutputInfo(networkInputs, networkOutputs, networkInputsCloned, networkOutputsCloned);

        ExecutableNetworkInternal::Ptr impl;
        if (nullptr == context) {
            impl = LoadExeNetworkImpl(network, config);
        } else {
            impl = LoadExeNetworkImpl(network, context, config);
        }

        impl->setNetworkInputs(networkInputsCloned);
        impl->setNetworkOutputs(networkOutputsCloned);
        impl->SetPointerToPlugin(shared_from_this());

        auto executableNetwork = make_executable_network(impl);
        return ExecutableNetwork(executableNetwork);
    }

protected:
    /**
     * @brief Creates an executable network from a parsed network object, users can create as many networks as they need
     *        and use them simultaneously (up to the limitation of the HW resources)
     * @note The function is used in
     * InferencePluginInternal::LoadNetwork(const ICNNNetwork&, const std::map<std::string, std::string>&)
     * which performs common steps first and calls this plugin-dependent method implementation after.
     * @param network A network object
     * @param config string-string map of config parameters relevant only for this load operation
     * @return Shared pointer to the ExecutableNetwork object
     */
    virtual ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const ICNNNetwork& network,
                                                              const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Creates an executable network using remote context from a parsed network object,
     * users can create as many networks as they need and use them simultaneously (up to the limitation of the HW resources)
     * @note The function is used in
     * InferencePluginInternal::LoadNetwork(const ICNNNetwork&, const std::map<std::string, std::string>&, RemoteContext::Ptr)
     * which performs common steps first and calls this plugin-dependent method implementation after.
     * @param network A network object
     * @param context A remote context
     * @param config string-string map of config parameters relevant only for this load operation
     * @return Shared pointer to the ExecutableNetwork object
     */
    virtual ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const ICNNNetwork& network,
                                                              RemoteContext::Ptr context,
                                                              const std::map<std::string, std::string>& config) {
        (void)network;
        (void)context;
        (void)config;
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    /**
     * @brief Creates an executable network from an previously exported network
     * @note The function is called from
     * IInferencePlugin::ImportNetwork(std::istream&, const RemoteContext::Ptr&, const std::map<std::string, std::string>&)
     * performs common steps first and calls this plugin-dependent implementation after.
     * @param networkModel Reference to network model output stream
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
                                                const std::map<std::string, std::string>& config) {
        (void)networkModel;
        (void)config;
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    /**
     * @brief Imports network wit RemoteContext
     * @param networkModel Reference to network model output stream
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
                                                const RemoteContext::Ptr& context,
                                                const std::map<std::string, std::string>& config) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    std::string _pluginName;  //!< A device name that plugins enables
    std::map<std::string, std::string> _config;  //!< A map config keys -> values
    ICore* _core = nullptr;  //!< A pointer to ICore interface
};

}  // namespace InferenceEngine
