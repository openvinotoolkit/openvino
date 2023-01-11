// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime plugin API wrapper, to be used by particular implementors
 * @file iplugin.hpp
 */

#pragma once

#include <memory>
#include <openvino/core/deprecated.hpp>

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "threading/ie_executor_manager.hpp"

namespace ov {

class ICore;
class IPlugin;
class CoreImpl;
class IInferencePluginWrapper;
class ICompiledModel;

namespace legacy_convert {

std::shared_ptr<::InferenceEngine::IInferencePlugin> convert_plugin(const std::shared_ptr<::ov::IPlugin>& plugin);
std::shared_ptr<::ov::IPlugin> convert_plugin(const std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin);

}  // namespace legacy_convert

}  // namespace ov

namespace InferenceEngine {

class IExtension;

}  // namespace InferenceEngine

namespace ov {

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
    void set_name(const std::string& name);

    /**
     * @brief Provides a plugin name
     *
     * @return Plugin name
     */
    const std::string& get_name() const;

    /**
     * @brief Compiles model from ov::Model object
     * @param model A model object acquired from ov::Core::read_model or source construction
     * @param properties A ov::AnyMap of properties relevant only for this load operation
     * @return Created Compiled Model object
     */
    std::shared_ptr<ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                  const ov::AnyMap& properties) const;

    /**
     * @brief Compiles model from ov::Model object, on specified remote context
     * @param model A model object acquired from ov::Core::read_model or source construction
     * @param properties A ov::AnyMap of properties relevant only for this load operation
     * @param context A pointer to plugin context derived from RemoteContext class used to
     *        execute the model
     * @return Created Compiled Model object
     */
    std::shared_ptr<ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                  const ov::AnyMap& properties,
                                                  const ov::RemoteContext& context) const;

    /**
     * @brief Sets properties for plugin, acceptable keys can be found in openvino/runtime/properties.hpp
     * @param properties ov::AnyMap of properties
     */
    virtual void set_property(const ov::AnyMap& properties);

    /**
     * @brief Gets properties related to plugin behaviour.
     *
     * @param name Property name.
     * @param arguments Additional arguments to get a property.
     *
     * @return Value of a property corresponding to the property name.
     */
    virtual ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const;

    /**
     * @brief Creates a remote context instance based on a map of properties
     * @param remote_properties Map of device-specific shared context remote properties.
     *
     * @return A remote context object
     */
    virtual RemoteContext create_context(const ov::AnyMap& remote_properties) const;

    /**
     * @brief Provides a default remote context instance if supported by a plugin
     * @param remote_properties Map of device-specific shared context remote properties.
     *
     * @return The default context.
     */
    virtual RemoteContext get_default_context(const ov::AnyMap& remote_properties) const;

    /**
     * @brief Creates an compiled model from an previously exported model using plugin implementation
     *        and removes OpenVINO Runtime magic and plugin name
     * @param model Reference to model output stream
     * @param properties A ov::AnyMap of properties
     * @return An Compiled model
     */
    virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const;

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
                                                             const ov::RemoteContext& context,
                                                             const ov::AnyMap& properties) const;

    /**
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    void set_core(std::weak_ptr<ov::ICore> core);

    /**
     * @brief Gets reference to ICore interface
     * @return Reference to ICore interface
     */
    std::shared_ptr<ov::ICore> get_core() const;

    /**
     * @brief Provides an information about used API
     * @return true if new API is used
     */
    bool is_new_api() const;

    /**
     * @brief Gets reference to tasks execution manager
     * @return Reference to ExecutorManager interface
     */
    const std::shared_ptr<InferenceEngine::ExecutorManager>& get_executor_manager() const;

    /**
     * @brief Queries a plugin about supported layers in model
     * @param model Model object to query.
     * @param properties Optional map of pairs: (property name, property value).
     * @return An object containing a map of pairs an operation name -> a device name supporting this operation.
     */
    virtual ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::AnyMap& properties) const;

    /**
     * @brief Registers legacy extension within plugin
     * @param extension - pointer to already loaded legacy extension
     */
    OPENVINO_DEPRECATED(
        "This method allows to load legacy InferenceEngine Extensions and will be removed in 2024.0 release")
    virtual void add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension);

    ~IPlugin() = default;

protected:
    IPlugin();

    /**
     * @brief Creates an compiled model from ov::Model object, users can create as many networks as they need
     *        and use them simultaneously (up to the limitation of the HW resources)
     * @note The function is used in
     * IPlugin::compile_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&)
     * which performs common steps first and calls this plugin-dependent method implementation after.
     * @param model A model object
     * @param properties ov::AnyMap of properties relevant only for this load operation
     * @return Shared pointer to the CompiledModel object
     */
    virtual std::shared_ptr<ICompiledModel> compile_model_impl(const std::shared_ptr<ov::Model>& model,
                                                               const ov::AnyMap& properties) const;

    /**
     * @brief Creates an compiled model from ov::Model object, users can create as many networks as they need
     *        and use them simultaneously (up to the limitation of the HW resources)
     * @note The function is used in
     * IPlugin::compile_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&, const ov::RemoteContext&)
     * which performs common steps first and calls this plugin-dependent method implementation after.
     * @param model A model object
     * @param context A remote context
     * @param properties ov::AnyMap of properties relevant only for this load operation
     * @return Shared pointer to the CompiledModel object
     */
    virtual std::shared_ptr<ICompiledModel> compile_model_impl(const std::shared_ptr<ov::Model>& model,
                                                               const ov::RemoteContext& context,
                                                               const ov::AnyMap& properties) const;

private:
    std::string m_plugin_name;                                             //!< A device name that plugins enables
    ov::AnyMap m_properties;                                               //!< A map config keys -> values
    std::weak_ptr<ov::ICore> m_core;                                       //!< A pointer to ICore interface
    std::shared_ptr<InferenceEngine::ExecutorManager> m_executor_manager;  //!< A tasks execution manager
    ov::Version m_version;                                                 //!< Member contains plugin version
    bool m_is_new_api;                                                     //!< A flag which shows used API
    std::shared_ptr<InferenceEngine::IInferencePlugin> old_plugin;
    friend ::ov::CoreImpl;
    friend ::ov::IInferencePluginWrapper;
    friend std::shared_ptr<::InferenceEngine::IInferencePlugin> ov::legacy_convert::convert_plugin(
        const std::shared_ptr<::ov::IPlugin>& plugin);
    friend std::shared_ptr<::ov::IPlugin> ov::legacy_convert::convert_plugin(
        const std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin);
    OPENVINO_DEPRECATED("Constructor is deprecated. Please do not use or re-implement it")
    IPlugin(const std::shared_ptr<InferenceEngine::IInferencePlugin>& ptr);
};

}  // namespace ov
/**
 * @def OV_CREATE_PLUGIN
 * @brief Defines a name of a function creating plugin instance
 * @ingroup ie_dev_api_plugin_api
 */
#ifndef OV_CREATE_PLUGIN
#    define OV_CREATE_PLUGIN CreatePluginEngine
#endif

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
        } catch (const InferenceEngine::Exception&) {                                                    \
            throw;                                                                                       \
        } catch (const std::exception& ex) {                                                             \
            IE_THROW() << ex.what();                                                                     \
        } catch (...) {                                                                                  \
            IE_THROW(Unexpected);                                                                        \
        }                                                                                                \
    }
