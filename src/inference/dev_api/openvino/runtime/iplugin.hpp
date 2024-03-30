// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime plugin API wrapper
 * @file openvino/runtime/iplugin.hpp
 */

#pragma once

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/util/pp.hpp"

namespace ov {

class ICompiledModel;

/**
 * @defgroup ov_dev_api OpenVINO Plugin API
 * @brief Defines OpenVINO Plugin API which can be used in plugin development
 *
 * @{
 * @defgroup ov_dev_api_plugin_api Plugin base classes
 * @brief A set of base and helper classes to implement a plugin class
 *
 * @defgroup ov_dev_api_compiled_model_api Compiled Model base classes
 * @brief A set of base and helper classes to implement an compiled model class
 *
 * @defgroup ov_dev_api_infer_request_api Inference Request common classes
 * @brief A set of base and helper classes to implement a common inference request functionality.
 *
 * @defgroup ov_dev_api_sync_infer_request_api Inference Request base classes
 * @brief A set of base and helper classes to implement a syncrhonous inference request class.
 *
 * @defgroup ov_dev_api_async_infer_request_api Asynchronous Inference Request base classes
 * @brief A set of base and helper classes to implement asynchronous inference request class
 *
 * @defgroup ov_dev_api_variable_state_api Variable state base classes
 * @brief A set of base and helper classes to implement variable state
 *
 * @defgroup ov_dev_api_threading Threading utilities
 * @brief Threading API providing task executors for asynchronous operations
 *
 * @defgroup ov_dev_api_system_conf System configuration utilities
 * @brief API to get information about the system, core processor capabilities
 *
 * @defgroup ov_dev_exec_model Execution model utilities
 * @brief Contains `ExecutionNode` and its properties
 *
 * @defgroup ov_dev_api_error_debug Error handling and debug helpers
 * @brief Utility methods to works with errors or exceptional situations
 *
 * @defgroup ov_dev_profiling ITT profiling utilities
 * @brief Configurable macro wrappers for ITT profiling
 *
 * @}
 */

/**
 * @brief OpenVINO Plugin Interface 2.0
 */
class OPENVINO_RUNTIME_API IPlugin : public std::enable_shared_from_this<IPlugin> {
public:
    /**
     * @brief Sets a plugin version
     *
     * @param version A version to set
     */
    void set_version(const Version& version);

    /**
     * @brief Returns a plugin version
     *
     * @return A constant ov::Version object
     */
    const Version& get_version() const;

    /**
     * @brief Sets a name for the plugin
     *
     * @param name Plugin name
     */
    void set_device_name(const std::string& name);

    /**
     * @brief Provides a plugin name
     *
     * @return Plugin name
     */
    const std::string& get_device_name() const;

    /**
     * @brief Compiles model from ov::Model object
     * @param model A model object acquired from ov::Core::read_model or source construction
     * @param properties A ov::AnyMap of properties relevant only for this load operation
     * @return Created Compiled Model object
     */
    virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties) const = 0;

    /**
     * @brief Compiles model from ov::Model object
     * @param model_path A path to model (path can be converted from unicode representation)
     * @param properties A ov::AnyMap of properties relevant only for this load operation
     * @return Created Compiled Model object
     */
    virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                              const ov::AnyMap& properties) const;

    /**
     * @brief Compiles model from ov::Model object, on specified remote context
     * @param model A model object acquired from ov::Core::read_model or source construction
     * @param properties A ov::AnyMap of properties relevant only for this load operation
     * @param context A pointer to plugin context derived from RemoteContext class used to
     *        execute the model
     * @return Created Compiled Model object
     */
    virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties,
                                                              const ov::SoPtr<ov::IRemoteContext>& context) const = 0;

    /**
     * @brief Sets properties for plugin, acceptable keys can be found in openvino/runtime/properties.hpp
     * @param properties ov::AnyMap of properties
     */
    virtual void set_property(const ov::AnyMap& properties) = 0;

    /**
     * @brief Gets properties related to plugin behaviour.
     *
     * @param name Property name.
     * @param arguments Additional arguments to get a property.
     *
     * @return Value of a property corresponding to the property name.
     */
    virtual ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const = 0;

    /**
     * @brief Creates a remote context instance based on a map of properties
     * @param remote_properties Map of device-specific shared context remote properties.
     *
     * @return A remote context object
     */
    virtual ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const = 0;

    /**
     * @brief Provides a default remote context instance if supported by a plugin
     * @param remote_properties Map of device-specific shared context remote properties.
     *
     * @return The default context.
     */
    virtual ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const = 0;

    /**
     * @brief Creates an compiled model from an previously exported model using plugin implementation
     *        and removes OpenVINO Runtime magic and plugin name
     * @param model Reference to model output stream
     * @param properties A ov::AnyMap of properties
     * @return An Compiled model
     */
    virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::AnyMap& properties) const = 0;

    /**
     * @brief Creates an compiled model from an previously exported model using plugin implementation
     *        and removes OpenVINO Runtime magic and plugin name
     * @param model Reference to model output stream
     * @param context A pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @param properties A ov::AnyMap of properties
     * @return An Compiled model
     */
    virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const = 0;

    /**
     * @brief Queries a plugin about supported layers in model
     * @param model Model object to query.
     * @param properties Optional map of pairs: (property name, property value).
     * @return An object containing a map of pairs an operation name -> a device name supporting this operation.
     */
    virtual ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::AnyMap& properties) const = 0;

    /**
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    void set_core(const std::weak_ptr<ov::ICore>& core);

    /**
     * @brief Gets reference to ICore interface
     * @return Reference to ICore interface
     */
    std::shared_ptr<ov::ICore> get_core() const;

    /**
     * @brief Gets reference to tasks execution manager
     * @return Reference to ExecutorManager interface
     */
    const std::shared_ptr<ov::threading::ExecutorManager>& get_executor_manager() const;

    virtual ~IPlugin() = default;

protected:
    IPlugin();

private:
    std::string m_plugin_name;                                           //!< A device name that plugins enables
    std::weak_ptr<ov::ICore> m_core;                                     //!< A pointer to ICore interface
    std::shared_ptr<ov::threading::ExecutorManager> m_executor_manager;  //!< A tasks execution manager
    ov::Version m_version;                                               //!< Member contains plugin version
};

/**
 * @brief Returns set of nodes from original model which are
 * determined as supported after applied transformation pipeline.
 * @param model Original model
 * @param transform Transformation pipeline function
 * @param is_node_supported Function returning whether node is supported or not
 * @param query_model_ratio The percentage of the model can be queried during query model (0 if not query)
 * @return Set of strings which contains supported node names
 */
OPENVINO_RUNTIME_API std::unordered_set<std::string> get_supported_nodes(
    const std::shared_ptr<const ov::Model>& model,
    std::function<void(std::shared_ptr<ov::Model>&)> transform,
    std::function<bool(const std::shared_ptr<ov::Node>)> is_node_supported,
    float query_model_ratio = 1.0f);

/**
 * @private
 */
using CreatePluginFunc = void(std::shared_ptr<::ov::IPlugin>&);

/**
 * @def OV_CREATE_PLUGIN
 * @brief Defines a name of a function creating plugin instance
 * @ingroup ov_dev_api_plugin_api
 */
#ifndef OV_CREATE_PLUGIN
#    define OV_CREATE_PLUGIN create_plugin_engine
#endif

/**
 * @private
 */
constexpr static const auto create_plugin_function = OV_PP_TOSTRING(OV_CREATE_PLUGIN);

}  // namespace ov

/**
 * @def OV_DEFINE_PLUGIN_CREATE_FUNCTION(PluginType, version)
 * @brief Defines the exported `OV_CREATE_PLUGIN` function which is used to create a plugin instance
 * @ingroup ov_dev_api_plugin_api
 */
#define OV_DEFINE_PLUGIN_CREATE_FUNCTION(PluginType, version, ...)                                       \
    OPENVINO_PLUGIN_API void OV_CREATE_PLUGIN(::std::shared_ptr<::ov::IPlugin>& plugin) noexcept(false); \
    void OV_CREATE_PLUGIN(::std::shared_ptr<::ov::IPlugin>& plugin) noexcept(false) {                    \
        try {                                                                                            \
            plugin = ::std::make_shared<PluginType>(__VA_ARGS__);                                        \
            plugin->set_version(version);                                                                \
        } catch (const std::exception& ex) {                                                             \
            OPENVINO_THROW(ex.what());                                                                   \
        }                                                                                                \
    }
