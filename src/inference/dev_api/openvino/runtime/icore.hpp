// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file provides API for Core object
 * @file openvino/runtime/icore.hpp
 */

#pragma once

#include <memory>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

namespace proxy {

class Plugin;

}

class ICompiledModel;
class IRemoteContext;

/**
 * @interface ICore
 * @brief Minimal ICore interface to allow plugin to get information from Core OpenVINO class.
 * @ingroup ov_dev_api_plugin_api
 */
class OPENVINO_RUNTIME_API ICore {
public:
    /**
     * @brief Reads IR xml and bin (with the same name) files
     * @param model string with IR
     * @param weights shared pointer to constant blob with weights
     * @param frontend_mode read network without post-processing or other transformations
     * @return shared pointer to ov::Model
     */
    virtual std::shared_ptr<ov::Model> read_model(const std::string& model,
                                                  const ov::Tensor& weights,
                                                  bool frontend_mode = false) const = 0;

    /**
     * @brief Reads IR xml and bin from buffer
     * @param model shared pointer to aligned buffer with IR
     * @param weights shared pointer to aligned buffer with weights
     * @return shared pointer to ov::Model
     */
    virtual std::shared_ptr<ov::Model> read_model(const std::shared_ptr<AlignedBuffer>& model,
                                                  const std::shared_ptr<AlignedBuffer>& weights) const = 0;

    /**
     * @brief Reads IR xml and bin files
     * @param model_path path to IR file
     * @param bin_path path to bin file, if path is empty, will try to read bin file with the same name as xml and
     * if bin file with the same name was not found, will load IR without weights.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this read operation.
     * @return shared pointer to ov::Model
     */
    virtual std::shared_ptr<ov::Model> read_model(const std::string& model_path,
                                                  const std::string& bin_path,
                                                  const AnyMap& properties) const = 0;

    virtual ov::AnyMap create_compile_config(const std::string& device_name, const ov::AnyMap& origConfig) const = 0;

    /**
     * @brief Creates a compiled mdel from a model object.
     *
     * Users can create as many models as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model OpenVINO Model
     * @param device_name Name of device to load model to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return A pointer to compiled model
     */
    virtual ov::SoPtr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                        const std::string& device_name,
                                                        const ov::AnyMap& config = {}) const = 0;

    /**
     * @brief Creates a compiled model from a model object.
     *
     * Users can create as many models as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model OpenVINO Model
     * @param context  "Remote" (non-CPU) accelerator device-specific execution context to use
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return A pointer to compiled model
     */
    virtual ov::SoPtr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                        const ov::SoPtr<ov::IRemoteContext>& context,
                                                        const ov::AnyMap& config = {}) const = 0;

    /**
     * @brief Creates a compiled model from a model file.
     *
     * Users can create as many models as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model_path Path to model
     * @param device_name Name of device to load model to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return A pointer to compiled model
     */
    virtual ov::SoPtr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                        const std::string& device_name,
                                                        const ov::AnyMap& config) const = 0;

    /**
     * @brief Creates a compiled model from a model memory.
     *
     * Users can create as many models as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param model_str String data of model
     * @param weights Model's weights
     * @param device_name Name of device to load model to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return A pointer to compiled model
     */
    virtual ov::SoPtr<ov::ICompiledModel> compile_model(const std::string& model_str,
                                                        const ov::Tensor& weights,
                                                        const std::string& device_name,
                                                        const ov::AnyMap& config) const = 0;

    /**
     * @brief Creates a compiled model from a previously exported model
     * @param model model stream
     * @param device_name Name of device load executable model on
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return A pointer to compiled model
     */
    virtual ov::SoPtr<ov::ICompiledModel> import_model(std::istream& model,
                                                       const std::string& device_name,
                                                       const ov::AnyMap& config = {}) const = 0;

    /**
     * @brief Creates a compiled model from a previously exported model
     * @param model model stream
     * @param context Remote context
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return A pointer to compiled model
     */
    virtual ov::SoPtr<ov::ICompiledModel> import_model(std::istream& modelStream,
                                                       const ov::SoPtr<ov::IRemoteContext>& context,
                                                       const ov::AnyMap& config = {}) const = 0;

    /**
     * @brief Query device if it supports specified network with specified configuration
     *
     * @param model OpenVINO Model
     * @param device_name A name of a device to query
     * @param config Optional map of pairs: (config parameter name, config parameter value)
     * @return An object containing a map of pairs a layer name -> a device name supporting this layer.
     */
    virtual ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                            const std::string& device_name,
                                            const ov::AnyMap& config) const = 0;

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    virtual std::vector<std::string> get_available_devices() const = 0;

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param device_name Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    virtual ov::SoPtr<ov::IRemoteContext> create_context(const std::string& device_name, const AnyMap& args) const = 0;

    /**
     * @brief Get a pointer to default shared context object for the specified device.
     * @param device_name  - A name of a device to get create shared context from.
     * @return A shared pointer to a default remote context.
     */
    virtual ov::SoPtr<ov::IRemoteContext> get_default_context(const std::string& device_name) const = 0;

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
     * @brief Get only properties that are supported by specified device
     * @param full_device_name Name of a device (can be either virtual or hardware)
     * @param properties Properties that can contains configs that are not supported by device
     * @param keep_core_property Whether to return core-level properties
     * @return map of properties that are supported by device
     */
    virtual AnyMap get_supported_property(const std::string& full_device_name, const AnyMap& properties, const bool keep_core_property = true) const = 0;

    virtual bool device_supports_model_caching(const std::string& device_name) const = 0;

    /**
     * @brief Default virtual destructor
     */
    virtual ~ICore();

private:
    virtual void set_property(const std::string& device_name, const AnyMap& properties) = 0;
    friend class ov::proxy::Plugin;
};

}  // namespace ov
