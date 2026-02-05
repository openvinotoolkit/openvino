// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache_guard.hpp"
#include "cache_manager.hpp"
#include "dev/plugin.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"

namespace ov {

using CreateExtensionFunc = void(std::vector<::ov::Extension::Ptr>&);
using CreatePluginEngineFunc = void(std::shared_ptr<::ov::IPlugin>&);

static const std::string default_device_name = "DEFAULT_DEVICE";
class CoreConfig final {
public:
    CoreConfig() = default;
    CoreConfig(const CoreConfig& other);
    CoreConfig& operator=(const CoreConfig&) = delete;

    struct CacheConfig {
        std::filesystem::path m_cache_dir;
        std::shared_ptr<ov::ICacheManager> m_cache_manager;

        static CacheConfig create(const std::filesystem::path& dir);
    };

    void set(const ov::AnyMap& config, const std::string& device_name);

    /**
     * @brief Removes core-level properties from config and triggers new state for core config
     * @param config      config to be updated
     * @param device_name device name for which config is applied (empty is for core-level)
     */
    void set_and_update(ov::AnyMap& config, const std::string& device_name);

    std::filesystem::path get_cache_dir() const;

    bool get_enable_mmap() const;

    // Creating thread-safe copy of global config including shared_ptr to ICacheManager
    CacheConfig get_cache_config_for_device(const ov::Plugin& plugin) const;

    // remove core properties
    static void remove_core(ov::AnyMap& config);

private:
    mutable std::mutex m_cache_config_mutex{};
    CacheConfig m_cache_config{};
    std::map<std::string, CacheConfig> m_devices_cache_config{};
    bool m_flag_enable_mmap{true};
};

struct Parsed {
    std::string m_device_name;
    AnyMap m_config;
    CoreConfig m_core_config;
};

/**
 * @brief Provides Parsed device name and configuration.
 *
 * Uses default core configuration updated with user properties from config.
 * The core properties are removed from user configuration for HW devices only.
 * @note The `CACHE_DIR` is not removed from compiled configuration.
 *
 * @param device_name                Device name to be parsed
 * @param config                    User configuration to be parsed.
 * @param keep_auto_batch_property  If set keep auto batch properties in compile properties.
 * @return Parsed:
 * - device name
 * - compile properties
 * - core configuration
 */
Parsed parse_device_name_into_config(const std::string& device_name,
                                     const AnyMap& config = {},
                                     const bool keep_auto_batch_property = false);

/**
 * @brief Provides Parsed device name and configuration.
 *
 * Uses user core configuration which is updated with user properties from config.
 * The core properties are removed from user configuration for HW devices only.
 * @note The `CACHE_DIR` is not removed from compiled configuration.
 *
 * @param device_name               Device name to be parsed
 * @param core_config               Core configuration used as base for parsed output.
 * @param config                    User configuration to be parsed.
 * @param keep_auto_batch_property  If set keep auto batch properties in compile properties.
 * @return Parsed:
 * - device name
 * - compile properties
 * - core configuration
 */
Parsed parse_device_name_into_config(const std::string& device_name,
                                     const CoreConfig& core_config,
                                     const AnyMap& config = {},
                                     const bool keep_auto_batch_property = false);

/**
 * @brief Checks whether config is applicable for device with 'device_name'
 * @code
 * core.compile_model(<device_name>, model, ov::device::properties(<device_name_to_parse>, ...));
 * @endcode
 * The common logic behind this is that 'device_name_to_parse' should match 'device_name' or be more
 * generic (e.g. GPU is more generic than GPU.x)
 *
 * @param device_name Target device
 * @param device_name_to_parse Device ID of property
 * @return true if ov::device::properties(<device_name_to_parse>, ...) is applicable for device identified by
 * 'device_name
 */
bool is_config_applicable(const std::string& device_name, const std::string& device_name_to_parse);

/**
 * @brief Checks whether the dvice is virtual device
 *
 * @param device_name Target device
 * @return true if target device is virtual device(e.g. AUTO, AUTO:XPU, AUTO:XPU.x, MULTI, MULTI:XPU, MULTI:XPU.x,
 * HETERO:XPU, HETERO:XPU.x, BATCH:XPU, BATCH:XPU.x)
 */
bool is_virtual_device(const std::string& device_name);

class CoreImpl : public ov::ICore, public std::enable_shared_from_this<ov::ICore> {
private:
    mutable std::map<std::string, ov::Plugin> m_plugins;
    // Mutex is needed to prevent changes of dev mutexes map from different threads
    mutable std::mutex m_global_mutex;
    // Global mutex "" locks parallel access to m_plugin_registry and plugins
    // Plugin mutexes "plugin_name" lock access to code which changes configuration of particular plugin
    mutable std::unordered_map<std::string, std::mutex> m_dev_mutexes;

    std::mutex& get_mutex(const std::string& dev_name = "") const;
    void add_mutex(const std::string& dev_name);

    bool is_proxy_device(const ov::Plugin& plugin) const;
    bool is_proxy_device(const std::string& dev_name) const;

    struct CacheContent {
        explicit CacheContent(const std::shared_ptr<ov::ICacheManager>& cache_manager,
                              bool mmap_enabled = false,
                              const std::filesystem::path model_path = {})
            : m_cache_manager(cache_manager),
              m_model_path(model_path),
              m_mmap_enabled{mmap_enabled} {}
        std::shared_ptr<ov::ICacheManager> m_cache_manager{};
        std::string m_blob_id{};
        std::filesystem::path m_model_path{};
        std::shared_ptr<const ov::Model> model{};
        bool m_mmap_enabled{};
    };

    // Core settings (cache config, etc)
    CoreConfig m_core_config;

    Any get_property_for_core(const std::string& name) const;

    mutable ov::CacheGuard m_cache_guard;

    struct PluginDescriptor {
        std::filesystem::path m_lib_location{};
        ov::AnyMap m_default_config{};
        std::vector<std::filesystem::path> m_list_of_extensions{};
        CreatePluginEngineFunc* m_plugin_create_func = nullptr;
        CreateExtensionFunc* m_extension_create_func = nullptr;
        mutable std::vector<Extension::Ptr> m_extensions{};  // mutable because of lazy init

        PluginDescriptor() = default;

        PluginDescriptor(const std::filesystem::path& lib_location,
                         const ov::AnyMap& default_config = {},
                         const std::vector<std::filesystem::path>& list_of_extensions = {})
            : m_lib_location(lib_location),
              m_default_config(default_config),
              m_list_of_extensions(list_of_extensions) {}

        PluginDescriptor(CreatePluginEngineFunc* plugin_create_func,
                         const ov::AnyMap& default_config = {},
                         CreateExtensionFunc* extension_create_func = nullptr)
            : m_lib_location(),
              m_default_config(default_config),
              m_list_of_extensions(),
              m_plugin_create_func(plugin_create_func),
              m_extension_create_func(extension_create_func) {}
    };

    std::shared_ptr<ov::threading::ExecutorManager> m_executor_manager;
    mutable std::unordered_set<std::string> m_opset_names;
    mutable std::vector<Extension::Ptr> m_extensions;
    std::map<std::string, PluginDescriptor> m_plugin_registry;

    ov::SoPtr<ov::ICompiledModel> compile_model_and_cache(ov::Plugin& plugin,
                                                          const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& parsed_config,
                                                          const ov::SoPtr<ov::IRemoteContext>& context,
                                                          const CacheContent& cache_content) const;

    ov::SoPtr<ov::ICompiledModel> load_model_from_cache(
        const CacheContent& cache_content,
        ov::Plugin& plugin,
        const ov::AnyMap& config,
        const ov::SoPtr<ov::IRemoteContext>& context,
        std::function<ov::SoPtr<ov::ICompiledModel>()> compile_model_lambda) const;

    bool device_supports_model_caching(const ov::Plugin& plugin, const ov::AnyMap& orig_config = {}) const;

    bool device_supports_property(const ov::Plugin& plugin, const ov::PropertyName& key) const;
    bool device_supports_internal_property(const ov::Plugin& plugin, const ov::PropertyName& key) const;

    ov::AnyMap create_compile_config(const ov::Plugin& plugin, const ov::AnyMap& orig_config) const;
    ov::AnyMap create_compile_config(const std::string& device_name, const ov::AnyMap& orig_config) const override {
        return create_compile_config(get_plugin(device_name), orig_config);
    }

    bool is_hidden_device(const std::string& device_name) const;
    void register_plugin_in_registry_unsafe(const std::string& device_name, PluginDescriptor& desc);


    void add_extensions_unsafe(const std::vector<ov::Extension::Ptr>& extensions) const;

    std::vector<ov::Extension::Ptr> get_extensions_copy() const;

public:
    CoreImpl();

    ~CoreImpl() override = default;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file.
     * @note The function supports UNICODE path
     * @param xml_config_file An .xml configuration with device / plugin information
     * @param by_abs_path A boolean value - register plugins by absolute file path or not
     */
    void register_plugins_in_registry(const std::filesystem::path& xml_config_file, const bool by_abs_path = false);

    std::shared_ptr<const ov::Model> apply_auto_batching(const std::shared_ptr<const ov::Model>& model,
                                                         std::string& device_name,
                                                         ov::AnyMap& config) const;

    /*
     * @brief Register plugins according to the build configuration
     */
    void register_compile_time_plugins();

    // Common API

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param plugin_name A name of device
     * @return Reference to a CPP plugin wrapper
     */
    ov::Plugin get_plugin(const std::string& plugin_name) const;

    /**
     * @brief Unload plugin for specified device, but plugin meta-data is still in plugin registry
     * @param device_name A name of device
     */
    void unload_plugin(const std::string& device_name);

    /**
     * @brief Registers plugin meta-data in registry for specified device
     * @param plugin Path (absolute or relative) or name of a plugin. Depending on platform `plugin` is wrapped with
     * shared library suffix and prefix to identify library full name
     * @param device_name A name of device
     * @param properties Plugin configuration
     */
    void register_plugin(const std::string& plugin, const std::string& device_name, const ov::AnyMap& properties);

    /**
     * @brief Provides a list of plugin names in registry; physically such plugins may not be created
     * @return A list of plugin names
     */
    std::vector<std::string> get_registered_devices() const;

    /**
     * @brief Sets config values for a plugin or set of plugins
     * @param device_name A device name to set config to
     *        If empty, config is set for all the plugins / plugin's meta-data
     * @note  `device_name` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
     *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
     */
    void set_property_for_device(const ov::AnyMap& config, const std::string& device_name);

    void add_extension(const std::vector<ov::Extension::Ptr>& extensions);

    bool device_supports_model_caching(const std::string& device_name) const override;

    // ov::ICore
    std::shared_ptr<ov::Model> read_model(const std::string& model,
                                          const ov::Tensor& weights,
                                          bool frontend_mode = false) const override;

    std::shared_ptr<ov::Model> read_model(const std::shared_ptr<AlignedBuffer>& model,
                                          const std::shared_ptr<AlignedBuffer>& weights) const override;

    std::shared_ptr<ov::Model> read_model(const std::string& model_path,
                                          const std::string& bin_path,
                                          const AnyMap& properties) const override;

    ov::SoPtr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                const std::string& device_name,
                                                const ov::AnyMap& config = {}) const override;

    ov::SoPtr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                const ov::SoPtr<ov::IRemoteContext>& context,
                                                const ov::AnyMap& config = {}) const override;

    ov::SoPtr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                const std::string& device_name,
                                                const ov::AnyMap& config) const override;

    ov::SoPtr<ov::ICompiledModel> compile_model(const std::string& model_str,
                                                const ov::Tensor& weights,
                                                const std::string& device_name,
                                                const ov::AnyMap& config) const override;

    ov::SoPtr<ov::ICompiledModel> import_model(std::istream& model,
                                               const std::string& device_name = {},
                                               const ov::AnyMap& config = {}) const override;

    ov::SoPtr<ov::ICompiledModel> import_model(std::istream& modelStream,
                                               const ov::SoPtr<ov::IRemoteContext>& context,
                                               const ov::AnyMap& config) const override;

    ov::SoPtr<ov::ICompiledModel> import_model(const ov::Tensor& compiled_blob,
                                               const std::string& device_name = {},
                                               const ov::AnyMap& config = {}) const override;

    ov::SoPtr<ov::ICompiledModel> import_model(const ov::Tensor& compiled_blob,
                                               const ov::SoPtr<ov::IRemoteContext>& context,
                                               const ov::AnyMap& config) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const std::string& device_name,
                                    const ov::AnyMap& config) const override;

    std::vector<std::string> get_available_devices() const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const std::string& device_name, const AnyMap& args) const override;

    ov::AnyMap get_supported_property(const std::string& device_name,
                                      const ov::AnyMap& config,
                                      const bool keep_core_property = true) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const std::string& device_name) const override;

    std::map<std::string, ov::Version> get_versions(const std::string& device_name) const;

    /**
     * @brief Sets properties for a device, acceptable keys can be found in openvino/runtime/properties.hpp.
     *
     * @param device_name Name of a device.
     *
     * @param properties Map of pairs: (property name, property value).
     */
    void set_property(const std::string& device_name, const AnyMap& properties) override;

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

    Any get_property(const std::string& device_name, const std::string& name, const AnyMap& arguments) const override;
};

}  // namespace ov
