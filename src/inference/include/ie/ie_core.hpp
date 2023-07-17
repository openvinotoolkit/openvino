// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine Core class C++ API
 *
 * @file ie_core.hpp
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp/ie_executable_network.hpp"
#include "ie_extension.h"
#include "ie_plugin_config.hpp"
#include "ie_remote_context.hpp"
#include "ie_version.hpp"

namespace InferenceEngine {

/**
 * @ingroup ie_cpp
 * @brief This class represents Inference Engine Core entity.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(Core) {
    class Impl;
    std::shared_ptr<Impl> _impl;

public:
    /** @brief Constructs an OpenVINO Core instance with devices
     * and their plugins description.
     *
     * There are two ways how to configure device plugins:
     * 1. (default) Use XML configuration file in case of dynamic libraries build;
     * 2. Use strictly defined configuration in case of static libraries build.
     *
     * @param xmlConfigFile Path to the .xml file with plugins to load from. If the XML configuration file is not
     * specified, default OpenVINO Runtime plugins are loaded from:
     * 1. (dynamic build) default `plugins.xml` file located in the same folder as OpenVINO runtime shared library;
     * 2. (static build) statically defined configuration. In this case path to the .xml file is ignored.
     */
    explicit Core(const std::string& xmlConfigFile = {});

    /**
     * @brief Returns plugins version information
     *
     * @param deviceName Device name to identify plugin
     * @return A vector of versions
     */
    std::map<std::string, Version> GetVersions(const std::string& deviceName) const;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Reads models from IR and ONNX formats
     * @param modelPath path to model
     * @param binPath path to data file
     * For IR format (*.bin):
     *  * if path is empty, will try to read bin file with the same name as xml and
     *  * if bin file with the same name was not found, will load IR without weights.
     * For ONNX format (*.onnx):
     *  * binPath parameter is not used.
     * @return CNNNetwork
     */
    CNNNetwork ReadNetwork(const std::wstring& modelPath, const std::wstring& binPath = {}) const;
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

    /**
     * @brief Reads models from IR and ONNX formats
     * @param modelPath path to model
     * @param binPath path to data file
     * For IR format (*.bin):
     *  * if path is empty, will try to read bin file with the same name as xml and
     *  * if bin file with the same name was not found, will load IR without weights.
     * For ONNX format (*.onnx):
     *  * binPath parameter is not used.
     * @return CNNNetwork
     */
    CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath = {}) const;

    /**
     * @brief Reads models from IR and ONNX formats
     * @param model string with model in IR or ONNX format
     * @param weights shared pointer to constant blob with weights
     * Reading ONNX models doesn't support loading weights from data blobs.
     * If you are using an ONNX model with external data files, please use the
     * `InferenceEngine::Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const`
     * function overload which takes a filesystem path to the model.
     * For ONNX case the second parameter should contain empty blob.
     * @note Created InferenceEngine::CNNNetwork object shares the weights with `weights` object.
     * So, do not create `weights` on temporary data which can be later freed, since the network
     * constant data becomes to point to invalid memory.
     * @return CNNNetwork
     */
    CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const;

    /**
     * @brief Creates an executable network from a network object and uses AUTO plugin as the default device to load
     * executable network.
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param network CNNNetwork object acquired from Core::ReadNetwork
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    ExecutableNetwork LoadNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config = {});

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
    ExecutableNetwork LoadNetwork(const CNNNetwork& network,
                                  const std::string& deviceName,
                                  const std::map<std::string, std::string>& config = {});

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file and uses AUTO plugin as the default
     * device to load executable network.
     *
     * This can be more efficient than using ReadNetwork + LoadNetwork(CNNNetwork) flow
     *        especially for cases when caching is enabled and cached model is available
     *
     * @param modelPath path to model
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation/
     *
     * @return An executable network reference
     */
    ExecutableNetwork LoadNetwork(const std::string& modelPath, const std::map<std::string, std::string>& config = {});

    /**
     * @brief Reads model and creates an executable network from IR or ONNX file
     *
     * This can be more efficient than using ReadNetwork + LoadNetwork(CNNNetwork) flow
     *        especially for cases when caching is enabled and cached model is available
     *
     * @param modelPath path to model
     * @param deviceName Name of device to load network to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation/
     *
     * @return An executable network reference
     */
    ExecutableNetwork LoadNetwork(const std::string& modelPath,
                                  const std::string& deviceName,
                                  const std::map<std::string, std::string>& config = {});

    /**
     * @brief Registers extension
     * @param extension Pointer to already loaded extension
     */
    void AddExtension(const IExtensionPtr& extension);

    /**
     * @brief Creates an executable network from a network object within a specified remote context.
     * @param network CNNNetwork object acquired from Core::ReadNetwork
     * @param context Pointer to RemoteContext object
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network object
     */
    ExecutableNetwork LoadNetwork(const CNNNetwork& network,
                                  RemoteContext::Ptr context,
                                  const std::map<std::string, std::string>& config = {});

    /**
     * @brief Registers extension for the specified plugin
     *
     * @param extension Pointer to already loaded extension
     * @param deviceName Device name to identify plugin to add an executable extension
     */
    void AddExtension(IExtensionPtr extension, const std::string& deviceName);

    /**
     * @brief Creates an executable network from a previously exported network
     *
     * @param modelFileName Path to the location of the exported file
     * @param deviceName Name of device load executable network on
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return An executable network reference
     */
    ExecutableNetwork ImportNetwork(const std::string& modelFileName,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config = {});

    /**
     * @brief Creates an executable network from a previously exported network
     * @param networkModel network model stream
     * @param deviceName Name of device load executable network on
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return An executable network reference
     */
    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config = {});

    /**
     * @deprecated Use Core::ImportNetwork with explicit device name
     * @brief Creates an executable network from a previously exported network
     * @param networkModel network model stream
     * @return An executable network reference
     */
    INFERENCE_ENGINE_DEPRECATED("Use Core::ImportNetwork with explicit device name")
    ExecutableNetwork ImportNetwork(std::istream& networkModel);

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
    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const RemoteContext::Ptr& context,
                                    const std::map<std::string, std::string>& config = {});

    /**
     * @brief Query device if it supports specified network with specified configuration
     *
     * @param deviceName A name of a device to query
     * @param network Network object to query
     * @param config Optional map of pairs: (config parameter name, config parameter value)
     * @return An object containing a map of pairs a layer name -> a device name supporting this layer.
     */
    QueryNetworkResult QueryNetwork(const CNNNetwork& network,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config = {}) const;

    /**
     * @brief Sets configuration for device, acceptable keys can be found in ie_plugin_config.hpp
     *
     * @param deviceName An optional name of a device. If device name is not specified, the config is set for all the
     * registered devices.
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName = {});

    /**
     * @brief Gets configuration dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via SetConfig method.
     *
     * @param deviceName  - A name of a device to get a configuration value.
     * @param name  - config key.
     * @return Value of config corresponding to config key.
     */
    Parameter GetConfig(const std::string& deviceName, const std::string& name) const;

    /**
     * @brief Gets general runtime metric for dedicated hardware.
     *
     * The method is needed to request common device properties
     * which are executable network agnostic. It can be device name, temperature, other devices-specific values.
     *
     * @param deviceName - A name of a device to get a metric value.
     * @param name - metric name to request.
     * @param options - optional parameters to get a metric value
     * @return Metric value corresponding to metric key.
     */
    Parameter GetMetric(const std::string& deviceName, const std::string& name, const ParamMap& options = {}) const;

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, GNA }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    std::vector<std::string> GetAvailableDevices() const;

    /**
     * @brief Register new device and plugin which implement this device inside Inference Engine.
     *
     * @param plugin Path (absolute or relative) or name of a plugin. Depending on platform, `plugin` is wrapped with
     * shared library suffix and prefix to identify library full name
     *
     * @param deviceName A device name to register plugin for
     */
    void RegisterPlugin(const std::string& plugin, const std::string& deviceName);

    /**
     * @brief Unloads previously loaded plugin with a specified name from Inference Engine
     * The method is needed to remove plugin instance and free its resources. If plugin for a
     * specified device has not been created before, the method throws an exception.
     *
     * @param deviceName Device name identifying plugin to remove from Inference Engine
     */
    void UnregisterPlugin(const std::string& deviceName);

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
     * - Properties are set to plugin via the `SetConfig` method.
     * - Extensions are set to plugin via the `AddExtension` method.
     *
     * @param xmlConfigFile A path to .xml file with plugins to register.
     */
    void RegisterPlugins(const std::string& xmlConfigFile);

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param deviceName Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    RemoteContext::Ptr CreateContext(const std::string& deviceName, const ParamMap& params);

    /**
     * @brief Get a pointer to default(plugin-supplied) shared context object for specified accelerator device.
     * @param deviceName  - A name of a device to get create shared context from.
     * @return A shared pointer to a default remote context.
     */
    RemoteContext::Ptr GetDefaultContext(const std::string& deviceName);
};

/**
 * @brief Shut down the OpenVINO by deleting all static-duration objects allocated by the library and releasing
 * dependent resources
 *
 * @note This function should be used by advanced user to control unload the resources.
 *
 * You might want to use this function if you are developing a dynamically-loaded library which should clean up all
 * resources after itself when the library is unloaded.
 */
INFERENCE_ENGINE_API_CPP(void) INFERENCE_ENGINE_1_0_DEPRECATED shutdown();
}  // namespace InferenceEngine
