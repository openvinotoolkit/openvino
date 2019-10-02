// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Inference Engine extended plugin API
 * \file ie_inference_plugin_api.hpp
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <ie_parameter.hpp>
#include <ie_api.h>

namespace InferenceEngine {

/**
 * @brief Forward declaration of ICorePrivate interface
 */
class ICore;

/**
 * @brief Extended plugin API to add new method to plugins but without changing public interface IInferencePlugin.
 * It should be used together with base IInferencePlugin which provides common interface, while this one just extends API.
 */
class IInferencePluginAPI {
public:
    /**
     * @brief Sets plugin name
     * @param pluginName Plugin name to set
     */
    virtual void SetName(const std::string & pluginName) noexcept = 0;

    /**
     * @brief Returns plugin name
     * @return Plugin name
     */
    virtual std::string GetName() const noexcept = 0;

    /**
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    virtual void SetCore(ICore *core) noexcept = 0;

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
    virtual Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter> & options) const = 0;

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  - metric name to request
     * @param options - configuration details for metric
     * @return Metric value corresponding to metric key
     */
    virtual Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter> & options) const = 0;

    virtual ~IInferencePluginAPI() = default;
};

class INFERENCE_ENGINE_API_CLASS(DeviceIDParser) {
    std::string deviceName;
    std::string deviceID;

public:
    explicit DeviceIDParser(const std::string & deviceNameWithID);

    std::string getDeviceID() const;
    std::string getDeviceName() const;

    static std::vector<std::string> getHeteroDevices(std::string fallbackDevice);
};

}  // namespace InferenceEngine
