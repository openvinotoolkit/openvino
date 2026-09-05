// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime plugin API wrapper
 * @file openvino/runtime/iplugin.hpp
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

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
     * @brief Compiles a model from a file path
     * @param model_path A path to model
     * @param properties A ov::AnyMap of properties relevant only for this load operation
     * @return Created Compiled Model object
     */
    virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::filesystem::path& model_path,
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
     * @param model Reference to model input stream
     * @param properties A ov::AnyMap of properties
     * @return An Compiled model
     */
    virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::AnyMap& properties) const = 0;

    /**
     * @brief Creates an compiled model from an previously exported model using plugin implementation
     *        and removes OpenVINO Runtime magic and plugin name
     * @param model Reference to model input stream
     * @param context A pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @param properties A ov::AnyMap of properties
     * @return An Compiled model
     */
    virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const = 0;

    /**
     * @brief Creates an compiled model from an previously exported model using plugin implementation
     *        and removes OpenVINO Runtime magic and plugin name
     * @param model Reference to ov::Tensor with exported model
     * @param properties A ov::AnyMap of properties
     * @return An Compiled model
     */
    virtual std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                             const ov::AnyMap& properties) const = 0;

    /**
     * @brief Creates an compiled model from an previously exported model using plugin implementation
     *        and removes OpenVINO Runtime magic and plugin name
     * @param model Reference to ov::Tensor with exported model
     * @param context A pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @param properties A ov::AnyMap of properties
     * @return An Compiled model
     */
    virtual std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
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

    /**
     * @brief Checks if a property is supported by the plugin.
     *
     * @param name Name of the property.
     * @param arguments Optional map of arguments for the property.
     * @return true if the property is supported, otherwise false.
     */
    virtual bool is_property_supported(const std::string& name, const ov::AnyMap& arguments = {}) const;

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

/**
 * @brief A score describing how well a plugin library can serve a physical device
 * during device-name dispatch. Higher wins; 0 means "cannot serve" and is excluded.
 * @ingroup ov_dev_api_plugin_api
 */
using DeviceCompatibilityScore = int32_t;

/// Vendor/tier mismatch: the library cannot serve this device. Never selected.
constexpr DeviceCompatibilityScore PROBE_SCORE_INCOMPATIBLE = 0;
/// Can run, but not preferred (fallback tier).
constexpr DeviceCompatibilityScore PROBE_SCORE_SERVABLE = 1;
/// Can run well and satisfies a hard requirement the peer library may not.
constexpr DeviceCompatibilityScore PROBE_SCORE_CAPABLE = 50;
/// Ideal runtime for this device.
constexpr DeviceCompatibilityScore PROBE_SCORE_PREFERRED = 100;

/**
 * @brief One physical device a plugin library reports it can serve, during
 * device-name dispatch. Produced by the enumeration probe (@ref EnumerateDevicesFunc),
 * consumed by ov::Core to reconcile candidate libraries for a shared device name.
 * @ingroup ov_dev_api_plugin_api
 */
struct EnumeratedDevice {
    /// The device id THIS library uses internally (".N"); may differ across libraries.
    std::string internal_id;
    /// Opaque cross-library identity token. Core compares it by equality only, never
    /// interprets it. Two libraries that build it over the same fields yield equal
    /// bytes for the same physical device.
    std::vector<uint8_t> fingerprint;
    /// How well this library serves the device (see PROBE_SCORE_* constants).
    DeviceCompatibilityScore score = PROBE_SCORE_INCOMPATIBLE;
};

/**
 * @private
 * @brief Signature of the enumeration probe. Enumerates every device this library can
 * serve, cheaply (a driver device query at most) and WITHOUT constructing the plugin
 * engine. noexcept, idempotent, no observable per-call side effect. It CLEARS the output
 * vector before filling it - it reports a device list, it does not append to one - so a
 * library that can serve nothing (no driver) leaves the vector empty.
 */
using EnumerateDevicesFunc = void(std::vector<EnumeratedDevice>& /*out*/) noexcept;

/**
 * @def OV_ENUMERATE_DEVICES
 * @brief Defines a name of a function running the device-dispatch enumeration probe. A static
 * build links every plugin into one binary, so the name is made per-device there (as for
 * OV_CREATE_PLUGIN) to keep the definitions from colliding.
 * @ingroup ov_dev_api_plugin_api
 */
#ifndef OV_ENUMERATE_DEVICES
#    define OV_ENUMERATE_DEVICES ov_enumerate_dispatch_devices
#endif

/**
 * @private
 */
constexpr static const auto enumerate_devices_function = OV_PP_TOSTRING(OV_ENUMERATE_DEVICES);

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

/**
 * @def OV_DEFINE_PLUGIN_ENUMERATE_FUNCTION(enumerate_fn)
 * @brief Defines the exported `ov_enumerate_dispatch_devices` device-dispatch probe by
 * forwarding to @p enumerate_fn (a callable with the ov::EnumerateDevicesFunc signature).
 * Used by plugins that participate in device-name dispatch (e.g. the GPU plugin).
 * The symbol is resolved and called by ov::Core BEFORE the plugin engine is constructed,
 * so @p enumerate_fn must not build an engine/context. See ov::EnumerateDevicesFunc.
 * @ingroup ov_dev_api_plugin_api
 */
#define OV_DEFINE_PLUGIN_ENUMERATE_FUNCTION(enumerate_fn)                                                   \
    OPENVINO_PLUGIN_API void OV_ENUMERATE_DEVICES(::std::vector<::ov::EnumeratedDevice>& devices) noexcept; \
    void OV_ENUMERATE_DEVICES(::std::vector<::ov::EnumeratedDevice>& devices) noexcept {                    \
        (enumerate_fn)(devices);                                                                            \
    }

/**
 * @def OV_DEFINE_PLUGIN_ENUMERATE_STUB()
 * @brief Defines the exported `ov_enumerate_dispatch_devices` probe as a no-op that reports
 * no devices. Used by every plugin that does NOT participate in device-name dispatch, so the
 * symbol is uniformly present across the plugin ABI. ov::Core never calls it for a
 * single-candidate device name, so a stubbed plugin keeps exactly today's behavior.
 * @ingroup ov_dev_api_plugin_api
 */
#define OV_DEFINE_PLUGIN_ENUMERATE_STUB()                                                                   \
    OPENVINO_PLUGIN_API void OV_ENUMERATE_DEVICES(::std::vector<::ov::EnumeratedDevice>& devices) noexcept; \
    void OV_ENUMERATE_DEVICES(::std::vector<::ov::EnumeratedDevice>& devices) noexcept {                    \
        devices.clear();                                                                                    \
    }
