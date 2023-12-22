// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for ICore interface
 * @file ie_icore.hpp
 */

#pragma once

#include <array>
#include <memory>
#include <string>

#include "cpp/ie_cnn_network.h"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "ie_parameter.hpp"
#include "ie_remote_context.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/properties.hpp"

namespace InferenceEngine {

class ICore : public ov::ICore {
public:
    /**
     * @brief Reads IR xml and bin (with the same name) files
     * @param model string with IR
     * @param weights shared pointer to constant blob with weights
     * @param frontendMode read network without post-processing or other transformations
     * @return CNNNetwork
     */
    virtual CNNNetwork ReadNetwork(const std::string& model,
                                   const Blob::CPtr& weights,
                                   bool frontendMode = false) const = 0;

    /**
     * @brief Reads IR xml and bin files
     * @param modelPath path to IR file
     * @param binPath path to bin file, if path is empty, will try to read bin file with the same name as xml and
     * if bin file with the same name was not found, will load IR without weights.
     * @return CNNNetwork
     */
    virtual CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const = 0;

    /**
     * @brief Creates an executable network from a network object.
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param network CNNNetwork object acquired from Core::ReadNetwork
     * @param deviceName Name of device to load network to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    virtual SoExecutableNetworkInternal LoadNetwork(const CNNNetwork& network,
                                                    const std::string& deviceName,
                                                    const std::map<std::string, std::string>& config = {}) = 0;

    /**
     * @brief Creates an executable network from a network object.
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param network CNNNetwork object acquired from Core::ReadNetwork
     * @param remoteCtx  "Remote" (non-CPU) accelerator device-specific execution context to use
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    virtual SoExecutableNetworkInternal LoadNetwork(const CNNNetwork& network,
                                                    const RemoteContext::Ptr& remoteCtx,
                                                    const std::map<std::string, std::string>& config = {}) = 0;

    /**
     * @brief Creates an executable network from a model memory.
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param modelStr String data of model
     * @param weights Model's weights
     * @param deviceName Name of device to load network to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @param val Optional callback to perform validation of loaded CNNNetwork, if ReadNetwork is triggered
     * @return An executable network reference
     */
    virtual SoExecutableNetworkInternal LoadNetwork(
        const std::string& modelStr,
        const InferenceEngine::Blob::CPtr& weights,
        const std::string& deviceName,
        const std::map<std::string, std::string>& config,
        const std::function<void(const InferenceEngine::CNNNetwork&)>& val = nullptr) = 0;

    /**
     * @brief Creates an executable network from a model file.
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param modelPath Path to model
     * @param deviceName Name of device to load network to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @param val Optional callback to perform validation of loaded CNNNetwork, if ReadNetwork is triggered
     * @return An executable network reference
     */
    virtual SoExecutableNetworkInternal LoadNetwork(const std::string& modelPath,
                                                    const std::string& deviceName,
                                                    const std::map<std::string, std::string>& config,
                                                    const std::function<void(const CNNNetwork&)>& val = nullptr) = 0;

    /**
     * @brief Creates an executable network from a previously exported network
     * @param networkModel network model stream
     * @param deviceName Name of device load executable network on
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return An executable network reference
     */
    virtual SoExecutableNetworkInternal ImportNetwork(std::istream& networkModel,
                                                      const std::string& deviceName = {},
                                                      const std::map<std::string, std::string>& config = {}) = 0;

    /**
     * @brief Query device if it supports specified network with specified configuration
     *
     * @param deviceName A name of a device to query
     * @param network Network object to query
     * @param config Optional map of pairs: (config parameter name, config parameter value)
     * @return An object containing a map of pairs a layer name -> a device name supporting this layer.
     */
    virtual QueryNetworkResult QueryNetwork(const CNNNetwork& network,
                                            const std::string& deviceName,
                                            const std::map<std::string, std::string>& config) const = 0;

    /**
     * @brief Gets general runtime metric for dedicated hardware.
     *
     * The method is needed to request common device properties
     * which are executable network agnostic. It can be device name, temperature, other devices-specific values.
     *
     * @param deviceName - A name of a device to get a metric value.
     * @param name - metric name to request.
     * @return Metric value corresponding to metric key.
     */
    virtual ov::Any GetMetric(const std::string& deviceName,
                              const std::string& name,
                              const ov::AnyMap& options = {}) const = 0;

    /**
     * @brief Gets configuration dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via SetConfig method.
     *
     * @param deviceName  - A name of a device to get a configuration value.
     * @param name  - config key.
     * @return Value of config corresponding to config key.
     */
    virtual ov::Any GetConfig(const std::string& deviceName, const std::string& name) const = 0;

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, GNA }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    virtual std::vector<std::string> GetAvailableDevices() const = 0;

    /**
     * @brief Checks whether device supports model caching feature
     *
     * @param deviceName - A name of a device to get a metric value.
     * @return True if device has IMPORT_EXPORT_SUPPORT and CACHING_PROPERTIES metric in SUPPORTED_METRICS and
     * this metric returns 'true', False otherwise.
     */
    virtual bool DeviceSupportsModelCaching(const std::string& deviceName) const = 0;

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param deviceName Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    virtual InferenceEngine::RemoteContext::Ptr CreateContext(const std::string& deviceName, const ov::AnyMap&) = 0;

    /**
     * @brief Get only configs that are supported by device
     * @param deviceName Name of a device
     * @param config Map of configs that can contains configs that are not supported by device
     * @return map of configs that are supported by device
     */
    virtual std::map<std::string, std::string> GetSupportedConfig(const std::string& deviceName,
                                                                  const std::map<std::string, std::string>& config) = 0;

    virtual bool isNewAPI() const = 0;

    /**
     * @brief Get a pointer to default shared context object for the specified device.
     * @param deviceName  - A name of a device to get create shared context from.
     * @return A shared pointer to a default remote context.
     */
    virtual RemoteContext::Ptr GetDefaultContext(const std::string& deviceName) = 0;
};

}  // namespace InferenceEngine
