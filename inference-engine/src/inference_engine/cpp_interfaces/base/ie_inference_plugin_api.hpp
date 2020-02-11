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

/**
 * @brief Forward declaration of ICorePrivate interface
 */
class ICore;

/**
 * @brief Extended plugin API to add new method to plugins but without changing public interface IInferencePlugin.
 * It should be used together with base IInferencePlugin which provides common interface, while this one just extends
 * API.
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

    virtual RemoteContext::Ptr CreateContext(const ParamMap& params) = 0;

    virtual RemoteContext::Ptr GetDefaultContext() = 0;

    /**
     * @brief Wraps original method
     * IInferencePlugin::ImportNetwork
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @return Created Executable Network object
     */
    virtual ExecutableNetwork LoadNetwork(ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                          RemoteContext::Ptr context) = 0;

    /**
     * @brief Wraps original method
     * IInferencePlugin::ImportNetwork
     * @param networkModel Network model input stream
     * @param config A configuration map
     * @return Created Executable Network object
     */
    virtual ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                            const std::map<std::string, std::string>& config) = 0;

    virtual ~IInferencePluginAPI();
};

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
