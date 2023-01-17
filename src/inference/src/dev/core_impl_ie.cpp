// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core_impl.hpp"
#include "dev/converter_utils.hpp"
#include "ie_network_reader.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/itt.hpp"

bool ov::CoreImpl::isNewAPI() const {
    return is_new_api();
}

InferenceEngine::RemoteContext::Ptr ov::CoreImpl::GetDefaultContext(const std::string& deviceName) {
    return get_default_context(deviceName)._impl;
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from file");
    return InferenceEngine::details::ReadNetwork(modelPath, binPath, extensions, ov_extensions, is_new_api());
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& model,
                                                      const InferenceEngine::Blob::CPtr& weights,
                                                      bool frontendMode) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from memory");
    return InferenceEngine::details::ReadNetwork(model, weights, extensions, ov_extensions, is_new_api(), frontendMode);
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
    ov::RemoteContext ctx{context, {nullptr}};
    auto compiled_model = compile_model(ov::legacy_convert::convert_model(network, isNewAPI()), ctx, any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::string& deviceNameOrig,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::CNN");
    auto compiled_model =
        compile_model(ov::legacy_convert::convert_model(network, isNewAPI()), deviceNameOrig, any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const std::string& modelPath,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    const std::function<void(const InferenceEngine::CNNNetwork&)>& val) {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::Path");
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto plugin = GetCPPPluginByName(parsed._deviceName);
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;
    auto conf = any_copy(parsed._config);
    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), conf)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager, modelPath};
    if (cacheManager && device_supports_import_export(plugin)) {
        bool loadedFromCache = false;
        cacheContent.blobId = CalculateFileHash(modelPath, parsed._deviceName, plugin, conf);
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, conf, {}, loadedFromCache);
        if (!loadedFromCache) {
            auto cnnNetwork = ReadNetwork(modelPath, std::string());
            if (val) {
                val(cnnNetwork);
            }
            res = compile_model_impl(ov::legacy_convert::convert_model(cnnNetwork, isNewAPI()),
                                     plugin,
                                     conf,
                                     {},
                                     cacheContent);
        }
    } else if (cacheManager) {
        res = plugin.compile_model(modelPath, conf);
    } else {
        auto cnnNetwork = ReadNetwork(modelPath, std::string());
        if (val) {
            val(cnnNetwork);
        }
        res = compile_model_impl(ov::legacy_convert::convert_model(cnnNetwork, isNewAPI()),
                                 plugin,
                                 conf,
                                 {},
                                 cacheContent);
    }
    return {res._ptr, res._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const std::string& modelStr,
    const InferenceEngine::Blob::CPtr& weights,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    const std::function<void(const InferenceEngine::CNNNetwork&)>& val) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::Memory");

    auto compiled_model = compile_model(modelStr,
                                        ov::Tensor{std::const_pointer_cast<InferenceEngine::Blob>(weights), {}},
                                        deviceName,
                                        ov::any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::ImportNetwork(
    std::istream& networkModel,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config) {
    auto compiled_model = import_model(networkModel, deviceName, any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::QueryNetworkResult ov::CoreImpl::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                               const std::string& deviceName,
                                                               const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::QueryNetwork");
    auto res = query_model(network.getFunction(), deviceName, any_copy(config));
    ie::QueryNetworkResult ret;
    if (!network.getFunction() || res.empty()) {
        ret.rc = InferenceEngine::GENERAL_ERROR;
        return ret;
    }
    ret.supportedLayersMap = res;

    const auto& func = network.getFunction();
    auto specialized_function = func->clone();

    std::string defDevice = ret.supportedLayersMap.begin()->second;
    ngraph::pass::ConstantFolding().run_on_model(specialized_function);
    std::unordered_set<std::string> opNames;

    for (const auto& op : specialized_function->get_ops())
        opNames.emplace(op->get_friendly_name());

    for (const auto& op : func->get_ops()) {
        if (opNames.find(op->get_friendly_name()) == opNames.end()) {
            ret.supportedLayersMap[op->get_friendly_name()] = defDevice;
        }
    }

    for (const auto& op : func->get_ops()) {
        if (!ret.supportedLayersMap.count(op->get_friendly_name()) &&
            std::dynamic_pointer_cast<ngraph::op::Constant>(op)) {
            bool are_all_users_supported = true;
            for (const auto& user : op->output(0).get_target_inputs()) {
                if (!ret.supportedLayersMap.count(user.get_node()->get_friendly_name())) {
                    are_all_users_supported = false;
                    break;
                }
            }
            if (are_all_users_supported) {
                ret.supportedLayersMap[op->get_friendly_name()] = defDevice;
            }
        }
    }
    return ret;
}

ov::Any ov::CoreImpl::GetMetric(const std::string& deviceName,
                                const std::string& name,
                                const ov::AnyMap& options) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            IE_THROW()
                << "You can get specific metrics with the GetMetric only for the HETERO itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            IE_THROW()
                << "You can get specific metrics with the GetMetric only for the MULTI itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // AUTO case
    {
        if (deviceName.find("AUTO:") == 0) {
            IE_THROW() << "You can get specific metrics with the GetMetric only for the AUTO itself (without devices). "
                          "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // BATCH case
    {
        if (deviceName.find("BATCH:") == 0) {
            IE_THROW()
                << "You can get specific metrics with the GetMetric only for the BATCH itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName);
    for (auto o : options) {
        parsed._config.insert(o);
    }

    return GetCPPPluginByName(parsed._deviceName).get_property(name, parsed._config);
}

ov::Any ov::CoreImpl::GetConfig(const std::string& deviceName, const std::string& name) const {
    auto parsed = parseDeviceNameIntoConfig(deviceName);
    return GetCPPPluginByName(parsed._deviceName).get_property(name, parsed._config);
}

std::vector<std::string> ov::CoreImpl::GetAvailableDevices() const {
    std::vector<std::string> devices;
    const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : GetListOfDevicesInRegistry()) {
        std::vector<std::string> devicesIDs;
        try {
            const InferenceEngine::Parameter p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string>>();
        } catch (const InferenceEngine::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const ov::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::runtime_error&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
            IE_THROW() << "An exception is thrown while trying to create the " << deviceName
                       << " device and call GetMetric: " << ex.what();
        } catch (...) {
            IE_THROW() << "Unknown exception is thrown while trying to create the " << deviceName
                       << " device and call GetMetric";
        }

        if (devicesIDs.size() > 1) {
            for (auto&& deviceID : devicesIDs) {
                devices.push_back(deviceName + '.' + deviceID);
            }
        } else if (!devicesIDs.empty()) {
            devices.push_back(deviceName);
        }
    }

    return devices;
}

InferenceEngine::RemoteContext::Ptr ov::CoreImpl::CreateContext(const std::string& deviceName,
                                                                const InferenceEngine::ParamMap& params) {
    return create_context(deviceName, params)._impl;
}

/**
 * @brief Registers the extension in a Core object
 *        Such extensions can be used for both CNNNetwork readers and device plugins
 */
void ov::CoreImpl::AddExtension(const InferenceEngine::IExtensionPtr& extension) {
    std::lock_guard<std::mutex> lock(get_mutex());
    AddExtensionUnsafe(extension);
}

bool ov::CoreImpl::DeviceSupportsImportExport(const std::string& deviceName) const {
    return device_supports_import_export(deviceName);
}
