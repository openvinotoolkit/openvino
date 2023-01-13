// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_extension.h>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_cache_guard.hpp"
#include "ie_cache_manager.hpp"
#include "ie_icore.hpp"
#include "multi-device/multi_device_config.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/runtime/common.hpp"
#include "threading/ie_executor_manager.hpp"

namespace ov {

const std::string DEFAULT_DEVICE_NAME = "DEFAULT_DEVICE";

template <typename T>
struct Parsed {
    std::string _deviceName;
    std::map<std::string, T> _config;
};

template <typename T = InferenceEngine::Parameter>
ov::Parsed<T> parseDeviceNameIntoConfig(const std::string& deviceName, const std::map<std::string, T>& config = {}) {
    auto config_ = config;
    auto deviceName_ = deviceName;
    if (deviceName_.find("HETERO:") == 0) {
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else if (deviceName_.find("MULTI:") == 0) {
        deviceName_ = "MULTI";
        config_[InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = deviceName.substr(6);
    } else if (deviceName == "AUTO" || deviceName.find("AUTO:") == 0) {
        deviceName_ = "AUTO";
        if (deviceName.find("AUTO:") == 0) {
            config_[InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] =
                deviceName.substr(std::string("AUTO:").size());
        }
    } else if (deviceName_.find("BATCH:") == 0) {
        deviceName_ = "BATCH";
        config_[CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG)] = deviceName.substr(6);
    } else {
        InferenceEngine::DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
            config_[InferenceEngine::PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }
    }
    return {deviceName_, config_};
}

class CoreImpl : public InferenceEngine::ICore, public std::enable_shared_from_this<InferenceEngine::ICore> {
private:
    mutable std::map<std::string, ov::InferencePlugin> plugins;
    // Mutex is needed to prevent changes of dev mutexes map from different threads
    mutable std::mutex global_mutex;
    // Global mutex "" locks parallel access to pluginRegistry and plugins
    // Plugin mutexes "plugin_name" lock access to code which changes configuration of particular plugin
    mutable std::unordered_map<std::string, std::mutex> dev_mutexes;

    std::mutex& get_mutex(const std::string& dev_name = "") const;
    void add_mutex(const std::string& dev_name);

    class CoreConfig final {
    public:
        struct CacheConfig {
            std::string _cacheDir;
            std::shared_ptr<InferenceEngine::ICacheManager> _cacheManager;
        };

        bool flag_allow_auto_batching = true;

        void setAndUpdate(ov::AnyMap& config);

        void setCacheForDevice(const std::string& dir, const std::string& name);

        std::string get_cache_dir() const;

        // Creating thread-safe copy of config including shared_ptr to ICacheManager
        // Passing empty or not-existing name will return global cache config
        CacheConfig getCacheConfigForDevice(const std::string& device_name,
                                            bool deviceSupportsCacheDir,
                                            std::map<std::string, std::string>& parsedConfig) const;

        CacheConfig getCacheConfigForDevice(const std::string& device_name) const;

    private:
        static void fillConfig(CacheConfig& config, const std::string& dir);

        mutable std::mutex _cacheConfigMutex;
        CacheConfig _cacheConfig;
        std::map<std::string, CacheConfig> _cacheConfigPerDevice;
    };

    struct CacheContent {
        explicit CacheContent(const std::shared_ptr<InferenceEngine::ICacheManager>& cache_manager,
                              const std::string model_path = {})
            : cacheManager(cache_manager),
              modelPath(model_path) {}
        std::shared_ptr<InferenceEngine::ICacheManager> cacheManager;
        std::string blobId = {};
        std::string modelPath = {};
    };

    // Core settings (cache config, etc)
    CoreConfig coreConfig;

    InferenceEngine::CacheGuard cacheGuard;

    struct PluginDescriptor {
        ov::util::FilePath libraryLocation;
        ov::AnyMap defaultConfig;
        std::vector<ov::util::FilePath> listOfExtentions;
        InferenceEngine::CreatePluginEngineFunc* pluginCreateFunc = nullptr;
        InferenceEngine::CreateExtensionFunc* extensionCreateFunc = nullptr;

        PluginDescriptor() = default;

        PluginDescriptor(const ov::util::FilePath& libraryLocation,
                         const ov::AnyMap& defaultConfig = {},
                         const std::vector<ov::util::FilePath>& listOfExtentions = {}) {
            this->libraryLocation = libraryLocation;
            this->defaultConfig = defaultConfig;
            this->listOfExtentions = listOfExtentions;
        }

        PluginDescriptor(InferenceEngine::CreatePluginEngineFunc* pluginCreateFunc,
                         const ov::AnyMap& defaultConfig = {},
                         InferenceEngine::CreateExtensionFunc* extensionCreateFunc = nullptr) {
            this->pluginCreateFunc = pluginCreateFunc;
            this->defaultConfig = defaultConfig;
            this->extensionCreateFunc = extensionCreateFunc;
        }
    };

    InferenceEngine::ExecutorManager::Ptr executorManagerPtr;
    mutable std::unordered_set<std::string> opsetNames;
    // TODO: make extensions to be optional with conditional compilation
    mutable std::vector<InferenceEngine::IExtensionPtr> extensions;
    std::vector<ov::Extension::Ptr> ov_extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;

    const bool newAPI;
    void AddExtensionUnsafe(const InferenceEngine::IExtensionPtr& extension) const;

    template <typename C, typename = FileUtils::enableIfSupportedChar<C>>
    void TryToRegisterLibraryAsExtensionUnsafe(const std::basic_string<C>& path) const {
        try {
            const auto extension_ptr = std::make_shared<InferenceEngine::Extension>(path);
            AddExtensionUnsafe(extension_ptr);
        } catch (const InferenceEngine::GeneralError&) {
            // in case of shared library is not opened
        }
    }

public:
    CoreImpl(bool _newAPI);

    ~CoreImpl() override = default;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file.
     * @note The function supports UNICODE path
     * @param xmlConfigFile An .xml configuraion with device / plugin information
     */
    void RegisterPluginsInRegistry(const std::string& xmlConfigFile);

#ifdef OPENVINO_STATIC_LIBRARY

    /**
     * @brief Register plugins for devices using statically defined configuration
     * @note The function supports UNICODE path
     * @param static_registry a statically defined configuration with device / plugin information
     */
    void RegisterPluginsInRegistry(const decltype(::getStaticPluginsRegistry())& static_registry);

#endif

    //
    // ICore public API
    //

    InferenceEngine::CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const override;

    InferenceEngine::CNNNetwork ReadNetwork(const std::string& model,
                                            const InferenceEngine::Blob::CPtr& weights,
                                            bool frontendMode = false) const override;

    bool isNewAPI() const override;

    static std::tuple<bool, std::string> CheckStatic(const InferenceEngine::CNNNetwork& network);

    InferenceEngine::RemoteContext::Ptr GetDefaultContext(const std::string& deviceName) override;

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override;

    void ApplyAutoBatching(const InferenceEngine::CNNNetwork& network,
                           std::string& deviceName,
                           std::map<std::string, std::string>& config);

    void CleanUpProperties(std::string& deviceName, std::map<std::string, std::string>& config, ov::Any property);

    InferenceEngine::SoExecutableNetworkInternal LoadNetwork(const InferenceEngine::CNNNetwork& network,
                                                             const std::string& deviceNameOrig,
                                                             const std::map<std::string, std::string>& config) override;

    InferenceEngine::SoExecutableNetworkInternal LoadNetwork(
        const std::string& modelPath,
        const std::string& deviceName,
        const std::map<std::string, std::string>& config,
        const std::function<void(const InferenceEngine::CNNNetwork&)>& val = nullptr) override;

    InferenceEngine::SoExecutableNetworkInternal LoadNetwork(
        const std::string& modelStr,
        const InferenceEngine::Blob::CPtr& weights,
        const std::string& deviceName,
        const std::map<std::string, std::string>& config,
        const std::function<void(const InferenceEngine::CNNNetwork&)>& val = nullptr) override;

    InferenceEngine::SoExecutableNetworkInternal ImportNetwork(
        std::istream& networkModel,
        const std::string& deviceName,
        const std::map<std::string, std::string>& config) override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::string& deviceName,
                                                     const std::map<std::string, std::string>& config) const override;

    Any GetMetric(const std::string& deviceName, const std::string& name, const AnyMap& options = {}) const override;

    void set_property(const std::string& device_name, const AnyMap& properties) override;

    Any get_property_for_core(const std::string& name) const;

    Any get_property(const std::string& device_name, const std::string& name, const AnyMap& arguments) const override;

    Any GetConfig(const std::string& deviceName, const std::string& name) const override;

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    std::vector<std::string> GetAvailableDevices() const override;

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param deviceName Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    InferenceEngine::RemoteContext::Ptr CreateContext(const std::string& deviceName,
                                                      const InferenceEngine::ParamMap& params) override;

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    ov::InferencePlugin GetCPPPluginByName(const std::string& pluginName) const;

    /**
     * @brief Unload plugin for specified device, but plugin meta-data is still in plugin registry
     * @param deviceName A name of device
     */
    void UnloadPluginByName(const std::string& deviceName);

    /**
     * @brief Registers plugin meta-data in registry for specified device
     * @param deviceName A name of device
     */
    void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName);

    /**
     * @brief Provides a list of plugin names in registry; physically such plugins may not be created
     * @return A list of plugin names
     */
    std::vector<std::string> GetListOfDevicesInRegistry() const;

    /**
     * @brief Sets config values for a plugin or set of plugins
     * @param deviceName A device name to set config to
     *        If empty, config is set for all the plugins / plugin's meta-data
     * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
     *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
     */
    void SetConfigForPlugins(const ov::AnyMap& configMap, const std::string& deviceName);

    /**
     * @brief Get device config it is passed as pair of device_name and `AnyMap`
     * @param configs All set of configs
     * @note  `device_name` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
     *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
     */
    void ExtractAndSetDeviceConfig(const ov::AnyMap& configs);

    std::map<std::string, std::string> GetSupportedConfig(const std::string& deviceName,
                                                          const std::map<std::string, std::string>& configs) override;

    /**
     * @brief Registers the extension in a Core object
     *        Such extensions can be used for both CNNNetwork readers and device plugins
     */
    void AddExtension(const InferenceEngine::IExtensionPtr& extension);

    void AddOVExtensions(const std::vector<ov::Extension::Ptr>& extensions);

    /**
     * @brief Provides a list of extensions
     * @return A list of registered extensions
     */
    const std::vector<InferenceEngine::IExtensionPtr>& GetExtensions() const;

    const std::vector<ov::Extension::Ptr>& GetOVExtensions() const;

    std::map<std::string, InferenceEngine::Version> GetVersions(const std::string& deviceName) const;

    bool DeviceSupportsImportExport(const std::string& deviceName) const override;

    bool DeviceSupportsConfigKey(const ov::InferencePlugin& plugin, const std::string& key) const;

    bool DeviceSupportsImportExport(const ov::InferencePlugin& plugin) const;

    bool DeviceSupportsCacheDir(const ov::InferencePlugin& plugin) const;

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> compile_model_impl(
        const InferenceEngine::CNNNetwork& network,
        ov::InferencePlugin& plugin,
        const std::map<std::string, std::string>& parsedConfig,
        const InferenceEngine::RemoteContext::Ptr& context,
        const CacheContent& cacheContent,
        bool forceDisableCache = false);

    static ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> LoadNetworkFromCache(
        const CacheContent& cacheContent,
        ov::InferencePlugin& plugin,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        bool& networkIsImported);

    std::map<std::string, std::string> CreateCompileConfig(const ov::InferencePlugin& plugin,
                                                           const std::string& deviceFamily,
                                                           const std::map<std::string, std::string>& origConfig) const;

    std::string CalculateNetworkHash(const InferenceEngine::CNNNetwork& network,
                                     const std::string& deviceFamily,
                                     const ov::InferencePlugin& plugin,
                                     const std::map<std::string, std::string>& config) const;

    std::string CalculateFileHash(const std::string& modelName,
                                  const std::string& deviceFamily,
                                  const ov::InferencePlugin& plugin,
                                  const std::map<std::string, std::string>& config) const;

    std::string CalculateMemoryHash(const std::string& modelStr,
                                    const ov::Tensor& weights,
                                    const std::string& deviceFamily,
                                    const ov::InferencePlugin& plugin,
                                    const std::map<std::string, std::string>& config) const;
};

}  // namespace ov
