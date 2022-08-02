// Copyright (C) 2018-2022 Intel Corporation
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
#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @interface ICore
 * @brief Minimal ICore interface to allow plugin to get information from Core Inference Engine class.
 * @ingroup ie_dev_api_plugin_api
 */
class ICore {
public:
    /**
     * @brief Reads IR xml and bin (with the same name) files
     * @param model string with IR
     * @param weights shared pointer to constant blob with weights
     * @param frontendMode read network without post-processing or other transformations
     * @return CNNNetwork
     */
    virtual ie::CNNNetwork ReadNetwork(const std::string& model,
                                       const ie::Blob::CPtr& weights,
                                       bool frontendMode = false) const = 0;

    /**
     * @brief Reads IR xml and bin files
     * @param modelPath path to IR file
     * @param binPath path to bin file, if path is empty, will try to read bin file with the same name as xml and
     * if bin file with the same name was not found, will load IR without weights.
     * @return CNNNetwork
     */
    virtual ie::CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const = 0;

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
    virtual ie::SoExecutableNetworkInternal LoadNetwork(const ie::CNNNetwork& network,
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
    virtual ie::SoExecutableNetworkInternal LoadNetwork(const ie::CNNNetwork& network,
                                                        const ie::RemoteContext::Ptr& remoteCtx,
                                                        const std::map<std::string, std::string>& config = {}) = 0;

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
    virtual ie::SoExecutableNetworkInternal LoadNetwork(
        const std::string& modelPath,
        const std::string& deviceName,
        const std::map<std::string, std::string>& config,
        const std::function<void(const ie::CNNNetwork&)>& val = nullptr) = 0;

    /**
     * @brief Creates an executable network from a previously exported network
     * @param networkModel network model stream
     * @param deviceName Name of device load executable network on
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return An executable network reference
     */
    virtual ie::SoExecutableNetworkInternal ImportNetwork(std::istream& networkModel,
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
    virtual ie::QueryNetworkResult QueryNetwork(const ie::CNNNetwork& network,
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
    virtual Any GetMetric(const std::string& deviceName, const std::string& name, const AnyMap& options = {}) const = 0;

    /**
     * @brief Gets configuration dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via SetConfig method.
     *
     * @param deviceName  - A name of a device to get a configuration value.
     * @param name  - config key.
     * @return Value of config corresponding to config key.
     */
    virtual Any GetConfig(const std::string& deviceName, const std::string& name) const = 0;

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    virtual std::vector<std::string> GetAvailableDevices() const = 0;

    /**
     * @brief Checks whether device supports Export & Import functionality of network
     *
     * @param deviceName - A name of a device to get a metric value.
     * @return True if device has IMPORT_EXPORT_SUPPORT metric in SUPPORTED_METRICS and
     * this metric returns 'true', False otherwise.
     */
    virtual bool DeviceSupportsImportExport(const std::string& deviceName) const = 0;

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param deviceName Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    virtual InferenceEngine::RemoteContext::Ptr CreateContext(const std::string& deviceName, const AnyMap&) = 0;

    /**
     * @brief Get only configs that are suppored by device
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
    virtual ie::RemoteContext::Ptr GetDefaultContext(const std::string& deviceName) = 0;

    /**
     * @brief Sets properties for a device, acceptable keys can be found in openvino/runtime/properties.hpp.
     *
     * @param device_name Name of a device.
     *
     * @param properties Map of pairs: (property name, property value).
     */
    virtual void set_property(const std::string& device_name, const AnyMap& properties) = 0;

    /**
     * @brief Sets properties for a device, acceptable keys can be found in openvino/runtime/properties.hpp.
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, Any>` types.
     * @param device_name Name of a device.
     * @param properties Optional pack of pairs: (property name, property value).
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(const std::string& device_name,
                                                                 Properties&&... properties) {
        set_property(device_name, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Gets properties related to device behaviour.
     *
     *
     * @param device_name  Name of a device to get a property value.
     * @param name  Property name.
     * @param arguments  Additional arguments to get a property.
     * @return Value of a property corresponding to the property name.
     */
    virtual Any get_property(const std::string& device_name,
                             const std::string& name,
                             const AnyMap& arguments) const = 0;

    /**
     * @brief Gets properties related to device behaviour.
     *
     * @tparam T Type of a returned value.
     * @tparam M Property mutability.
     * @param deviceName  Name of a device to get a property value.
     * @param property  Property object.
     * @return Property value.
     */
    template <typename T, PropertyMutability M>
    T get_property(const std::string& device_name, const Property<T, M>& property) const {
        return get_property(device_name, property.name(), {}).template as<T>();
    }

    /**
     * @brief Gets properties related to device behaviour.
     *
     * @tparam T Type of a returned value.
     * @tparam M Property mutability.
     * @param deviceName  Name of a device to get a property value.
     * @param property  Property object.
     * @param arguments  Additional arguments to get a property.
     * @return Property value.
     */
    template <typename T, PropertyMutability M>
    T get_property(const std::string& device_name, const Property<T, M>& property, const AnyMap& arguments) const {
        return get_property(device_name, property.name(), arguments).template as<T>();
    }

    /**
     * @brief Default virtual destructor
     */
    virtual ~ICore() = default;
};
}  // namespace ov

namespace InferenceEngine {
using ICore = ov::ICore;
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
    static std::string getBatchDevice(std::string devicesList);
};

}  // namespace InferenceEngine
