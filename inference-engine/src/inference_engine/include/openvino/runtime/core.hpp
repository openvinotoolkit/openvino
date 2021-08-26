// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime Core class C++ API
 *
 * @file openvino/runtime/core.hpp
 */
#pragma once

#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "cpp/ie_executable_network.hpp"
#include "ie_plugin_config.hpp"
#include "ie_version.hpp"
#include "remote_context.hpp"

namespace InferenceEngine {
class IExtension;
class Blob;
class RemoteContext;
}  // namespace InferenceEngine

namespace ov {

class Function;

namespace runtime {

/**
 * @brief This class represents OpenVINO runtime Core entity.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class INFERENCE_ENGINE_API_CLASS(Core) {
    class Impl;
    std::shared_ptr<Impl> _impl;

public:
    /** @brief Constructs OpenVINO Core instance using XML configuration file with
     * plugins description.
     *
     * See register_plugins for more details.
     *
     * @param xmlConfigFile A path to .xml file with plugins to load from. If XML configuration file is not specified,
     * then default Inference Engine plugins are loaded from the default plugin.xml file.
     */
    explicit Core(const std::string& xmlConfigFile = {});

    /**
     * @brief Returns plugins version information
     *
     * @param deviceName Device name to identify plugin
     * @return A vector of versions
     */
    std::map<std::string, ie::Version> get_versions(const std::string& deviceName) const;

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Reads models from IR and ONNX formats
     * @param modelPath path to model
     * @param binPath path to data file
     * For IR format (*.bin):
     *  * if path is empty, will try to read bin file with the same name as xml and
     *  * if bin file with the same name was not found, will load IR without weights.
     * For ONNX format (*.onnx or *.prototxt):
     *  * binPath parameter is not used.
     * @return Function
     */
    std::shared_ptr<ov::Function> read_model(const std::wstring& modelPath, const std::wstring& binPath = {}) const;
#endif

    /**
     * @brief Reads models from IR and ONNX formats
     * @param modelPath path to model
     * @param binPath path to data file
     * For IR format (*.bin):
     *  * if path is empty, will try to read bin file with the same name as xml and
     *  * if bin file with the same name was not found, will load IR without weights.
     * For ONNX format (*.onnx or *.prototxt):
     *  * binPath parameter is not used.
     * @return Function
     */
    std::shared_ptr<ov::Function> read_model(const std::string& modelPath, const std::string& binPath = {}) const;
    /**
     * @brief Reads models from IR and ONNX formats
     * @param model string with model in IR or ONNX format
     * @param weights shared pointer to constant blob with weights
     * Reading ONNX models doesn't support loading weights from data blobs.
     * If you are using an ONNX model with external data files, please use the
     * `ov::runtime::Core::read_model(const std::string& model, const Blob::CPtr& weights) const`
     * function overload which takes a filesystem path to the model.
     * For ONNX case the second parameter should contain empty blob.
     * @note Created Function object shares the weights with `weights` object.
     * So, do not create `weights` on temporary data which can be later freed, since the network
     * constant data becomes to point to invalid memory.
     * @return Function
     */
    std::shared_ptr<ov::Function> read_model(const std::string& model,
                                             const std::shared_ptr<const ie::Blob>& weights) const;

    /**
     * @brief Creates an executable network from a network object.
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param network Function object acquired from Core::read_model
     * @param deviceName Name of device to load network to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    ie::ExecutableNetwork compile_model(const std::shared_ptr<const ov::Function>& network,
                                        const std::string& deviceName,
                                        const ConfigMap& config = {});

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file
     *
     * This can be more efficient than using read_model + compile_model(Function) flow
     *        especially for cases when caching is enabled and cached model is available
     *
     * @param modelPath path to model
     * @param deviceName Name of device to load network to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation/
     *
     * @return An executable network reference
     */
    ie::ExecutableNetwork compile_model(const std::string& modelPath,
                                        const std::string& deviceName,
                                        const ConfigMap& config = {});

    /**
     * @brief Creates an executable network from a network object within a specified remote context.
     * @param network Function object acquired from Core::read_model
     * @param context Pointer to RemoteContext object
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network object
     */
    ie::ExecutableNetwork compile_model(const std::shared_ptr<const ov::Function>& network,
                                        const RemoteContext& context,
                                        const ConfigMap& config = {});

    /**
     * @brief Registers extension
     * @param extension Pointer to already loaded extension
     */
    void add_extension(const std::shared_ptr<ie::IExtension>& extension);

    /**
     * @brief Creates an executable network from a previously exported network
     * @param networkModel network model stream
     * @param deviceName Name of device load executable network on
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return An executable network reference
     */
    ie::ExecutableNetwork import_model(std::istream& networkModel,
                                       const std::string& deviceName,
                                       const ConfigMap& config = {});

    /**
     * @brief Creates an executable network from a previously exported network within a specified
     * remote context.
     *
     * @param networkModel Network model stream
     * @param context Pointer to RemoteContext object
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    ie::ExecutableNetwork import_model(std::istream& networkModel,
                                       const RemoteContext& context,
                                       const ConfigMap& config = {});

    /**
     * @brief Query device if it supports specified network with specified configuration
     *
     * @param deviceName A name of a device to query
     * @param network Network object to query
     * @param config Optional map of pairs: (config parameter name, config parameter value)
     * @return An object containing a map of pairs a layer name -> a device name supporting this layer.
     */
    ie::QueryNetworkResult query_model(const std::shared_ptr<const ov::Function>& network,
                                       const std::string& deviceName,
                                       const ConfigMap& config = {}) const;

    /**
     * @brief Sets configuration for device, acceptable keys can be found in ie_plugin_config.hpp
     *
     * @param deviceName An optional name of a device. If device name is not specified, the config is set for all the
     * registered devices.
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void set_config(const ConfigMap& config, const std::string& deviceName = {});

    /**
     * @brief Gets configuration dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via set_config method.
     *
     * @param deviceName  - A name of a device to get a configuration value.
     * @param name  - config key.
     * @return Value of config corresponding to config key.
     */
    ie::Parameter get_config(const std::string& deviceName, const std::string& name) const;

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
    ie::Parameter get_metric(const std::string& deviceName, const std::string& name) const;

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, FPGA.0, FPGA.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    std::vector<std::string> get_available_devices() const;

    /**
     * @brief Register new device and plugin which implement this device inside Inference Engine.
     *
     * @param pluginName A name of plugin. Depending on platform pluginName is wrapped with shared library suffix and
     * prefix to identify library full name
     *
     * @param deviceName A device name to register plugin for. If device name is not specified, then it's taken from
     * plugin itself.
     */
    void register_plugin(const std::string& pluginName, const std::string& deviceName);

    /**
     * @brief Unloads previously loaded plugin with a specified name from Inference Engine
     * The method is needed to remove plugin instance and free its resources. If plugin for a
     * specified device has not been created before, the method throws an exception.
     *
     * @param deviceName Device name identifying plugin to remove from Inference Engine
     */
    void unload_plugin(const std::string& deviceName);

    /** @brief Registers plugin to Inference Engine Core instance using XML configuration file with
     * plugins description.
     *
     *  XML file has the following structure:
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
     * - `name` identifies name of device enabled by plugin
     * - `location` specifies absolute path to dynamic library with plugin. A path can also be relative to inference
     * engine shared library. It allows to have common config for different systems with different configurations.
     * - Properties are set to plugin via the `set_config` method.
     * - Extensions are set to plugin via the `add_extension` method.
     *
     * @param xmlConfigFile A path to .xml file with plugins to register.
     */
    void register_plugins(const std::string& xmlConfigFile);

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param deviceName Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    RemoteContext create_context(const std::string& deviceName, const ie::ParamMap& params);

    /**
     * @brief Get a pointer to default(plugin-supplied) shared context object for specified accelerator device.
     * @param deviceName  - A name of a device to get create shared context from.
     * @return A shared pointer to a default remote context.
     */
    RemoteContext get_default_context(const std::string& deviceName);
};
}  // namespace runtime
}  // namespace ov
