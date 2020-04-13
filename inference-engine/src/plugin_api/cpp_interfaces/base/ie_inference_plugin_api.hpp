// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Inference Engine extended plugin API
 * \file ie_inference_plugin_api.hpp
 */

#pragma once

#include <ie_api.h>

#include <cpp/ie_executable_network.hpp>
#include <ie_parameter.hpp>
#include <ie_remote_context.hpp>
#include <map>
#include <string>
#include <vector>

namespace InferenceEngine {

class ICore;

/**
 * @brief Extends Inference Engine Plugin API to add new method to plugins but without changing the public IInferencePlugin interface.
 * It should be used together with base IInferencePlugin which provides common interface, while this one just extends API.
 * @ingroup ie_dev_api_plugin_api
 */
class INFERENCE_ENGINE_API_CLASS(IInferencePluginAPI) {
public:
    /**
     * @brief Sets plugin name
     * @param pluginName Plugin name to set
     */
    virtual void SetName(const std::string& pluginName) noexcept = 0;

    /**
     * @brief Returns plugin name
     * @return Plugin name
     */
    virtual std::string GetName() const noexcept = 0;

    /**
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    virtual void SetCore(ICore* core) noexcept = 0;

    /**
     * @brief Gets refernce to ICore interface
     * @return Reference to core interface
     */
    virtual const ICore& GetCore() const = 0;

    /**
     * @brief Gets configuration dedicated to plugin behaviour
     * @param name  - value of config corresponding to config key
     * @param options - configuration details for config
     * @return Value of config corresponding to config key
     */
    virtual Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const = 0;

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  - metric name to request
     * @param options - configuration details for metric
     * @return Metric value corresponding to metric key
     */
    virtual Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const = 0;

    /**
     * @brief      Creates a remote context instance based on a map of parameters
     * @param[in]  params  The map of parameters
     * @return     A remote context object
     */
    virtual RemoteContext::Ptr CreateContext(const ParamMap& params) = 0;

    /**
     * @brief      Provides a default remote context instance if supported by a plugin
     * @return     The default context.
     */
    virtual RemoteContext::Ptr GetDefaultContext() = 0;

    /**
     * @brief Wraps original method
     * IInferencePlugin::LoadNetwork
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @return Created Executable Network object
     */
    virtual ExecutableNetwork LoadNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                          RemoteContext::Ptr context) = 0;

    /**
     * @brief Creates an executable network from an previously exported network using plugin implementation
     *        and removes Inference Engine magic and plugin name
     * @param networkModel Reference to network model output stream
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                            const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Creates an executable network from an previously exported network using plugin implementation
     *        and removes Inference Engine magic and plugin name
     * @param networkModel Reference to network model output stream
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                            const RemoteContext::Ptr& context,
                                            const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief A virtual descturctor
     */
    virtual ~IInferencePluginAPI();
};

/**
 * @private
 */
class INFERENCE_ENGINE_API_CLASS(DeviceIDParser) {
    std::string deviceName;
    std::string deviceID;
public:
    explicit DeviceIDParser(const std::string& deviceNameWithID);

    std::string getDeviceID() const;
    std::string getDeviceName() const;

    static std::vector<std::string> getHeteroDevices(std::string fallbackDevice);
    static std::vector<std::string> getMultiDevices(std::string devicesList);
};

}  // namespace InferenceEngine
