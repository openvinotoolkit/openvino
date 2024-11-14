// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache_guard.hpp"
#include "dev/plugin.hpp"
#include "cache_manager.hpp"
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

const std::string DEFAULT_DEVICE_NAME = "DEFAULT_DEVICE";

struct Parsed {
    std::string _deviceName;
    AnyMap _config;
};

Parsed parseDeviceNameIntoConfig(const std::string& deviceName,
                                 const AnyMap& config = {},
                                 const bool keep_core_property = false);

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

class CoreImpl : public ov::ICore, public std::enable_shared_from_this<ov::ICore> {
private:
    mutable std::map<std::string, ov::Plugin> plugins;
    // Mutex is needed to prevent changes of dev mutexes map from different threads
    mutable std::mutex global_mutex;
    // Global mutex "" locks parallel access to pluginRegistry and plugins
    // Plugin mutexes "plugin_name" lock access to code which changes configuration of particular plugin
    mutable std::unordered_map<std::string, std::mutex> dev_mutexes;

    std::mutex& get_mutex(const std::string& dev_name = "") const;
    void add_mutex(const std::string& dev_name);

    bool is_proxy_device(const ov::Plugin& plugin) const;
    bool is_proxy_device(const std::string& dev_name) const;

    class CoreConfig final {
    public:
        struct CacheConfig {
            std::string _cacheDir;
            std::shared_ptr<ov::ICacheManager> _cacheManager;

            static CacheConfig create(const std::string& dir);
        };

        /**
         * @brief Removes core-level properties from config and triggers new state for core config
         * @param config - config to be updated
         */
        void set_and_update(ov::AnyMap& config);

        OPENVINO_DEPRECATED("Don't use this method, it will be removed soon")
        void set_cache_dir_for_device(const std::string& dir, const std::string& name);

        std::string get_cache_dir() const;

        bool get_enable_mmap() const;

        // Creating thread-safe copy of config including shared_ptr to ICacheManager
        // Passing empty or not-existing name will return global cache config
        CacheConfig get_cache_config_for_device(const ov::Plugin& plugin, ov::AnyMap& parsedConfig) const;

    private:
        mutable std::mutex _cacheConfigMutex;
        CacheConfig _cacheConfig;
        std::map<std::string, CacheConfig> _cacheConfigPerDevice;
        bool _flag_enable_mmap = true;
    };

    struct CacheContent {
        explicit CacheContent(const std::shared_ptr<ov::ICacheManager>& cache_manager,
                              const std::string model_path = {})
            : cacheManager(cache_manager),
              modelPath(model_path) {}
        std::shared_ptr<ov::ICacheManager> cacheManager;
        std::string blobId = {};
        std::string modelPath = {};
    };

    // Core settings (cache config, etc)
    CoreConfig coreConfig;

    Any get_property_for_core(const std::string& name) const;

    mutable ov::CacheGuard cacheGuard;

    struct PluginDescriptor {
        ov::util::FilePath libraryLocation;
        ov::AnyMap defaultConfig;
        std::vector<ov::util::FilePath> listOfExtentions;
        CreatePluginEngineFunc* pluginCreateFunc = nullptr;
        CreateExtensionFunc* extensionCreateFunc = nullptr;

        PluginDescriptor() = default;

        PluginDescriptor(const ov::util::FilePath& libraryLocation,
                         const ov::AnyMap& defaultConfig = {},
                         const std::vector<ov::util::FilePath>& listOfExtentions = {}) {
            this->libraryLocation = libraryLocation;
            this->defaultConfig = defaultConfig;
            this->listOfExtentions = listOfExtentions;
        }

        PluginDescriptor(CreatePluginEngineFunc* pluginCreateFunc,
                         const ov::AnyMap& defaultConfig = {},
                         CreateExtensionFunc* extensionCreateFunc = nullptr) {
            this->pluginCreateFunc = pluginCreateFunc;
            this->defaultConfig = defaultConfig;
            this->extensionCreateFunc = extensionCreateFunc;
        }
    };

    std::shared_ptr<ov::threading::ExecutorManager> m_executor_manager;
    mutable std::unordered_set<std::string> opsetNames;
    mutable std::vector<ov::Extension::Ptr> extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;

    ov::SoPtr<ov::ICompiledModel> compile_model_and_cache(ov::Plugin& plugin,
                                                          const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& parsedConfig,
                                                          const ov::SoPtr<ov::IRemoteContext>& context,
                                                          const CacheContent& cacheContent) const;

    ov::SoPtr<ov::ICompiledModel> load_model_from_cache(
        const CacheContent& cacheContent,
        ov::Plugin& plugin,
        const ov::AnyMap& config,
        const ov::SoPtr<ov::IRemoteContext>& context,
        std::function<ov::SoPtr<ov::ICompiledModel>()> compile_model_lambda) const;

    bool device_supports_model_caching(const ov::Plugin& plugin) const;

    bool device_supports_property(const ov::Plugin& plugin, const ov::PropertyName& key) const;
    bool device_supports_internal_property(const ov::Plugin& plugin, const ov::PropertyName& key) const;

    OPENVINO_DEPRECATED("Don't use this method, it will be removed soon")
    bool device_supports_cache_dir(const ov::Plugin& plugin) const;

    ov::AnyMap create_compile_config(const ov::Plugin& plugin, const ov::AnyMap& origConfig) const;
    ov::AnyMap create_compile_config(const std::string& device_name, const ov::AnyMap& origConfig) const override {
        return create_compile_config(get_plugin(device_name), origConfig);
    }

    bool is_hidden_device(const std::string& device_name) const;
    void register_plugin_in_registry_unsafe(const std::string& device_name, PluginDescriptor& desc);

    template <typename C, typename = ov::util::enableIfSupportedChar<C>>
    void try_to_register_plugin_extensions(const std::basic_string<C>& path) const {
        try {
            auto plugin_extensions = ov::detail::load_extensions(path);
            add_extensions_unsafe(plugin_extensions);
        } catch (const std::runtime_error&) {
            // in case of shared library is not opened
        }
    }
    void add_extensions_unsafe(const std::vector<ov::Extension::Ptr>& extensions) const;

public:
    CoreImpl();

    ~CoreImpl() override = default;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file.
     * @note The function supports UNICODE path
     * @param xml_config_file An .xml configuraion with device / plugin information
     * @param by_abs_path A boolean value - register plugins by absolute file path or not
     */
    void register_plugins_in_registry(const std::string& xml_config_file, const bool& by_abs_path = false);

    std::shared_ptr<const ov::Model> apply_auto_batching(const std::shared_ptr<const ov::Model>& model,
                                                         std::string& deviceName,
                                                         ov::AnyMap& config) const;

    /*
     * @brief Register plugins according to the build configuration
     */
    void register_compile_time_plugins();

    // Common API

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param pluginName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    ov::Plugin get_plugin(const std::string& pluginName) const;

    /**
     * @brief Unload plugin for specified device, but plugin meta-data is still in plugin registry
     * @param deviceName A name of device
     */
    void unload_plugin(const std::string& deviceName);

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
     * @param deviceName A device name to set config to
     *        If empty, config is set for all the plugins / plugin's meta-data
     * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
     *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
     */
    void set_property_for_device(const ov::AnyMap& configMap, const std::string& deviceName);

    void add_extension(const std::vector<ov::Extension::Ptr>& extensions);

    bool device_supports_model_caching(const std::string& device_name) const override;

    // ov::ICore
    std::shared_ptr<ov::Model> read_model(const std::string& model,
                                          const ov::Tensor& weights,
                                          bool frontend_mode = false) const override;

    std::shared_ptr<ov::Model> read_model(const std::shared_ptr<AlignedBuffer>& model,
                                          const std::shared_ptr<AlignedBuffer>& weights) const override;

    std::shared_ptr<ov::Model> read_model(const std::string& model_path, const std::string& bin_path) const override;

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

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const std::string& device_name,
                                    const ov::AnyMap& config) const override;

    std::vector<std::string> get_available_devices() const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const std::string& device_name, const AnyMap& args) const override;

    ov::AnyMap get_supported_property(const std::string& device_name, const ov::AnyMap& config, const bool keep_core_property = true) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const std::string& device_name) const override;

    std::map<std::string, ov::Version> get_versions(const std::string& deviceName) const;

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
