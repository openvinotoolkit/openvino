// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for ICore interface
 * @file icore.hpp
 */

#pragma once

#include <memory>
#include <openvino/core/any.hpp>
#include <openvino/core/model.hpp>
#include <openvino/runtime/common.hpp>
#include <openvino/runtime/compiled_model.hpp>

namespace ov {

/**
 * @interface ICore
 * @brief Minimal ICore interface to allow plugin to get information from OpenVINO Core class.
 * @ingroup ie_dev_api_plugin_api
 */
class ICore {
public:
    /**
     * @brief Reads model from the memory
     * @param model string with model in framework format
     * @param weights shared pointer to constant tensor with weights
     * @param frontendMode read network without post-processing or other transformations
     * @return Shared pointer to ov::Model
     */
    virtual std::shared_ptr<ov::Model> read_model(const std::string& model,
                                                  const ov::Tensor& weights,
                                                  bool frontendMode = false) const = 0;

    /**
     * @brief Reads model from the files
     * @param modelPath path to model file
     * @param binPath path to weight file, if path is empty, for IR will try to read bin file with the same name as xml
     * and if bin file with the same name was not found, will load IR without weights.
     * @return Shared pointer to ov::Model
     */
    virtual std::shared_ptr<ov::Model> read_model(const std::string& modelPath, const std::string& binPath) const = 0;

    /**
     * @brief Creates an compiled model from the model object
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model ov::Model object acquired from Core::read_model
     * @param deviceName Name of device to load network to
     * @param properties Optional map of properties
     * @return An compile model
     */
    virtual ov::CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model,
                                            const std::string& deviceName,
                                            const ov::AnyMap& properties = {}) = 0;

    /**
     * @brief Creates an compiled model from the model object
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model ov::Model object acquired from Core::read_model
     * @param remoteCtx  "Remote" (non-CPU) accelerator device-specific execution context to use
     * @param properties Optional map of properties
     * @return An compile model
     */
    virtual ov::CompiledModel compile_model(const std::shared_ptr<const ov::Model>& network,
                                            const ov::RemoteContext& remoteCtx,
                                            const ov::AnyMap& config = {}) = 0;

    /**
     * @brief Creates an executable network from a previously exported network
     * @param model stream with model
     * @param deviceName Name of device load executable network on
     * @param properties Optional map of properties
     * @return An compile model
     */
    virtual ov::CompiledModel import_model(std::istream& model,
                                           const std::string& deviceName = {},
                                           const ov::AnyMap& config = {}) = 0;

    /**
     * @brief Query device if it supports specified network with specified configuration
     *
     * @param model Model object to query
     * @param deviceName A name of a device to query
     * @param properties Optional map of properties
     * @return An compile model
     */
    virtual ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                            const std::string& deviceName,
                                            const ov::AnyMap& config) const = 0;

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    virtual std::vector<std::string> get_available_devices() const = 0;

    /**
     * @brief Checks whether device supports Export & Import functionality of network
     *
     * @param deviceName - A name of a device to get a metric value.
     * @return True if device has IMPORT_EXPORT_SUPPORT metric in SUPPORTED_METRICS and
     * this metric returns 'true', False otherwise.
     */
    virtual bool device_supports_import_export(const std::string& deviceName) const = 0;

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param deviceName Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A created remote context.
     */
    virtual ov::RemoteContext create_context(const std::string& deviceName, const AnyMap&) = 0;

    /**
     * @brief Get only configs that are suppored by device
     * @param deviceName Name of a device
     * @param config Map of configs that can contains configs that are not supported by device
     * @return map of configs that are supported by device
     */
    virtual ov::AnyMap GetSupportedConfig(const std::string& deviceName, const ov::AnyMap& config) = 0;

    /**
     * @brief Allow to understand what API is used for the inference
     *
     * @return true if OV 2.0 API is used
     */
    virtual bool is_new_api() const = 0;

    /**
     * @brief Get a default shared context object for the specified device.
     * @param deviceName  - A name of a device to get create shared context from.
     * @return A default remote context.
     */
    virtual ov::RemoteContext get_default_context(const std::string& deviceName) = 0;

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
