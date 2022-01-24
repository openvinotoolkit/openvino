// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the OpenVINO Runtime Core class C++ API.
 *
 * @file openvino/runtime/core.hpp
 */
#pragma once

#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_plugin_config.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/op.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/tensor.hpp"

namespace InferenceEngine {
class IExtension;
}  // namespace InferenceEngine

namespace ov {

namespace runtime {

/**
 * @brief This class represents an OpenVINO runtime Core entity.
 * User applications can create several Core class instances, but in this case the underlying plugins
 * are created multiple times and not shared between several Core instances. The recommended way is to have
 * a single Core instance per application.
 */
class OPENVINO_RUNTIME_API Core {
    class Impl;
    std::shared_ptr<Impl> _impl;

public:
    /** @brief Constructs an OpenVINO Core instance using the XML configuration file with
     * devices and their plugins description.
     *
     * See Core::register_plugins for more details.
     *
     * @param xml_config_file Path to the .xml file with plugins to load from. If the XML configuration file is not specified,
     * default OpenVINO Runtime plugins are loaded from the default `plugin.xml` file located in the same folder
     * as OpenVINO runtime shared library.
     */
    explicit Core(const std::string& xml_config_file = {});

    /**
     * @brief Returns device plugins version information.
     * Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
     * in this case, std::map contains multiple entries, each per device.
     *
     * @param device_name Device name to identify a plugin.
     * @return A vector of versions.
     */
    std::map<std::string, Version> get_versions(const std::string& device_name) const;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Reads models from IR/ONNX/PDPD formats.
     * @param model_path Path to a model.
     * @param bin_path Path to a data file.
     * For IR format (*.bin):
     *  * if path is empty, will try to read a bin file with the same name as xml and
     *  * if the bin file with the same name is not found, will load IR without weights.
     * For ONNX format (*.onnx):
     *  * the bin_path parameter is not used.
     * For PDPD format (*.pdmodel)
     *  * the bin_path parameter is not used.
     * @return A model.
     */
    std::shared_ptr<ov::Model> read_model(const std::wstring& model_path, const std::wstring& bin_path = {}) const;
#endif

    /**
     * @brief Reads models from IR/ONNX/PDPD formats.
     * @param model_path Path to a model.
     * @param bin_path Path to a data file.
     * For IR format (*.bin):
     *  * if path is empty, will try to read a bin file with the same name as xml and
     *  * if the bin file with the same name is not found, will load IR without weights.
     * For ONNX format (*.onnx):
     *  * the bin_path parameter is not used.
     * For PDPD format (*.pdmodel)
     *  * the bin_path parameter is not used.
     * @return A model.
     */
    std::shared_ptr<ov::Model> read_model(const std::string& model_path, const std::string& bin_path = {}) const;

    /**
     * @brief Reads models from IR/ONNX/PDPD formats.
     * @param model String with a model in IR/ONNX/PDPD format.
     * @param weights Shared pointer to a constant tensor with weights.
     * Reading ONNX/PDPD models does not support loading weights from the @p weights tensors.
     * @note Created model object shares the weights with the @p weights object.
     * Thus, do not create @p weights on temporary data that can be freed later, since the model
     * constant data will point to an invalid memory.
     * @return A model.
     */
    std::shared_ptr<ov::Model> read_model(const std::string& model, const Tensor& weights) const;

    /**
     * @brief Creates and loads a compiled model from a source model to the default OpenVINO device selected by the AUTO
     * plugin.
     *
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources).
     *
     * @param model Model object acquired from Core::read_model.
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model, const ConfigMap& config = {});

    /**
     * @brief Creates a compiled model from a source model object.
     *
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources).
     *
     * @param model Model object acquired from Core::read_model.
     * @param device_name Name of a device to load a model to.
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model,
                                const std::string& device_name,
                                const ConfigMap& config = {});

    /**
     * @brief Reads and loads a compiled model from the IR/ONNX/PDPD file to the default OpenVINO device selected by the
     * AUTO plugin.
     *
     * This can be more efficient than using the Core::read_model + Core::compile_model(model_in_memory_object) flow,
     * especially for cases when caching is enabled and a cached model is available.
     *
     * @param model_path Path to a model.
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation.
     *
     * @return A compiled model.
     */
    CompiledModel compile_model(const std::string& model_path, const ConfigMap& config = {});

    /**
     * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
     *
     * This can be more efficient than using the Core::read_model + Core::compile_model(model_in_memory_object) flow,
     * especially for cases when caching is enabled and a cached model is available.
     *
     * @param model_path Path to a model.
     * @param device_name Name of a device to load a model to.
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation.
     *
     * @return A compiled model.
     */
    CompiledModel compile_model(const std::string& model_path,
                                const std::string& device_name,
                                const ConfigMap& config = {});

    /**
     * @brief Creates a compiled model from a source model within a specified remote context.
     * @param model Model object acquired from Core::read_model.
     * @param context A reference to a RemoteContext object.
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation.
     * @return A compiled model object.
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model,
                                const RemoteContext& context,
                                const ConfigMap& config = {});

    /**
     * @deprecated This method is deprecated. Please use other Core::add_extension methods.
     * @brief Registers OpenVINO 1.0 extension to a Core object.
     * @param extension Pointer to the already loaded extension.
     */
    OPENVINO_DEPRECATED("Please use add_extension(ov::Extension) or add_extension(path_to_library) instead.")
    void add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension);

    /**
     * @brief Registers an extension to a Core object.
     * @param library_path Path to the library with ov::Extension.
     */
    void add_extension(const std::string& library_path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Registers an extension to a Core object.
     * @param library_path Unicode path to the library with ov::Extension.
     */
    void add_extension(const std::wstring& library_path);
#endif

    /**
     * @brief Registers an extension to a Core object.
     * @param extension Pointer to the extension.
     */
    void add_extension(const std::shared_ptr<ov::Extension>& extension);

    /**
     * @brief Registers extensions to a Core object.
     * @param extensions Vector of loaded extensions.
     */
    void add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions);

    /**
     * @brief Registers an extension to a Core object.
     * @param extension Extension class that is inherited from the ov::Extension class.
     */
    template <class T, typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
    }

    /**
     * @brief Registers extensions to a Core object.
     * @param extension Extension class that is inherited from the ov::Extension class.
     * @param args A list of extensions.
     */
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension, Targs... args) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
        add_extension(args...);
    }

    /**
     * @brief Registers a custom operation inherited from ov::op::Op.
     */
    template <class T, typename std::enable_if<std::is_base_of<ov::op::Op, T>::value, bool>::type = true>
    void add_extension() {
        std::shared_ptr<ov::Extension> ext = std::make_shared<ov::OpExtension<T>>();
        add_extension(ext);
    }

    /**
     * @brief Registers custom operations inherited from ov::op::Op.
     */
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::op::Op, T>::value && sizeof...(Targs), bool>::type = true>
    void add_extension() {
        std::shared_ptr<ov::Extension> ext = std::make_shared<ov::OpExtension<T>>();
        add_extension(ext);
        if (sizeof...(Targs) > 0)
            add_extension<Targs...>();
    }

    /**
     * @brief Imports a compiled model from the previously exported one.
     * @param model_stream std::istream input stream containing a model previously exported using the
     * ov::runtime::CompiledModel::export_model method.
     * @param device_name Name of a device to import a compiled model for. Note, if the @p device_name device was not used to
     * compile the original mode, an exception is thrown.
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel import_model(std::istream& model_stream,
                               const std::string& device_name,
                               const ConfigMap& config = {});

    /**
     * @brief Imports a compiled model from the previously exported one with the specified remote context.
     * @param model_stream std::istream input stream containing a model previously exported from
     * ov::runtime::CompiledModel::export_model.
     * @param context A reference to a RemoteContext object. Note, if the device from @p context was not used to compile
     * the original mode, an exception is thrown.
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel import_model(std::istream& model_stream, const RemoteContext& context, const ConfigMap& config = {});

    /**
     * @brief Query device if it supports the specified model with specified configuration.
     *
     * @param device_name A name of a device to query.
     * @param model Model object to query.
     * @param config Optional map of pairs: (config parameter name, config parameter value).
     * @return An object containing a map of pairs an operation name -> a device name supporting this operation.
     */
    SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                const std::string& device_name,
                                const ConfigMap& config = {}) const;

    /**
     * @brief Sets configuration for a device; acceptable keys can be found in ie_plugin_config.hpp.
     *
     * @param device_name An optional name of a device. If device name is not specified, the config is set for all the
     * registered devices.
     *
     * @param config Map of pairs: (config parameter name, config parameter value).
     */
    void set_config(const ConfigMap& config, const std::string& device_name = {});

    /**
     * @brief Gets configuration dedicated to device behaviour.
     * The method is targeted to extract information that can be set via the Core::set_config method.
     *
     * @param device_name Name of a device to get a configuration value.
     * @param config_key_name Config key name.
     * @return Value of a config corresponding to the config key.
     */
    Any get_config(const std::string& device_name, const std::string& config_key_name) const;

    /**
     * @brief Gets general runtime metric for dedicated hardware.
     *
     * The method is needed to request common device or system properties.
     * It can be device name, temperature, and other devices-specific values.
     *
     * @param device_name Name of a device to get a metric value.
     * @param metric_key_name Metric name to request.
     * @return Metric value corresponding to the metric key.
     */
    Any get_metric(const std::string& device_name, const std::string& metric_key_name) const;

    /**
     * @brief Returns devices available for inference.
     * Core objects go over all registered plugins and ask about available devices.
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }.
     * If there is more than one device of a specific type, they are enumerated with the .# suffix.
     * Such enumerated device can later be used as a device name in all Core methods like Core::compile_model,
     * Core::query_model, Core::set_config, and so on.
     */
    std::vector<std::string> get_available_devices() const;

    /**
     * @brief Register a new device and plugin that enables this device inside OpenVINO Runtime.
     *
     * @param plugin_name Name of a plugin. Depending on platform, `plugin_name` is wrapped with shared library suffix
     * and prefix to identify library full name.
     * For example, on Linux platform, plugin name specified as `plugin_name` will be wrapped as `libplugin_name.so`.
     * Plugin search algorithm:
     * - If plugin is located in the same directory as OpenVINO runtime library, it will be used.
     * - If no, plugin is tried to be loaded from paths pointed by PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
     *   environment variables depending on the platform.
     *
     * @param device_name Device name to register a plugin for.
     */
    void register_plugin(const std::string& plugin_name, const std::string& device_name);

    /**
     * @brief Unloads the previously loaded plugin identified by @p device_name from OpenVINO Runtime.
     * The method is needed to remove loaded plugin instance and free its resources. If plugin for a
     * specified device has not been created before, the method throws an exception.
     * @note This method does not remove plugin from the plugins known to OpenVINO Core object.
     * @param device_name Device name identifying plugin to remove from OpenVINO Runtime.
     */
    void unload_plugin(const std::string& device_name);

    /** @brief Registers a device plugin to the OpenVINO Runtime Core instance using an XML configuration file with
     * plugins description.
     *
     *  The XML file has the following structure:
     *
     * ```xml
     * <ie>
     *     <plugins>
     *         <plugin name="" location="">
     *             <extensions>
     *                 <extension location=""/>
     *             </extensions>
     *             <properties>
     *                 <property key="" value=""/>
     *             </properties>
     *         </plugin>
     *     </plugins>
     * </ie>
     * ```
     *
     * - `name` identifies name of a device enabled by a plugin.
     * - `location` specifies absolute path to dynamic library with a plugin.
     *    The path can also be relative to inference engine shared library. It allows having common config
     *    for different systems with different configurations.
     * - `properties` are set to a plugin via the ov::runtime::Core::set_config method.
     * - `extensions` are set to a plugin via the ov::runtime::Core::add_extension method.
     *
     * @param xml_config_file A path to .xml file with plugins to register.
     */
    void register_plugins(const std::string& xml_config_file);

    /**
     * @brief Creates a new remote shared context object on the specified accelerator device
     * using specified plugin-specific low-level device API parameters (device handle, pointer, context, etc.).
     * @param device_name Name of a device to create a new shared context on.
     * @param params A map of device-specific shared context parameters.
     * @return Reference to a created remote context.
     */
    RemoteContext create_context(const std::string& device_name, const ParamMap& params);

    /**
     * @brief Gets a pointer to default (plugin-supplied) shared context object for the specified accelerator device.
     * @param device_name Name of a device to get a default shared context from.
     * @return Reference to a default remote context.
     */
    RemoteContext get_default_context(const std::string& device_name);
};
}  // namespace runtime
}  // namespace ov
