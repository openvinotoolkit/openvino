// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <tuple>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <legacy/ie_util_internal.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/parsed_config.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/error.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>

#include "generic_ie.hpp"

#include "myriad_plugin.h"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::VPUConfigParams;
using namespace vpu::MyriadPlugin;


ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(LoadExeNetworkImpl);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    return std::make_shared<ExecutableNetwork>(network, _mvnc, _devicePool, parsedConfigCopy, GetCore());
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    _parsedConfig.update(config);

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    auto supported_keys = _metrics->SupportedConfigKeys();
    if (std::find(supported_keys.begin(),
        supported_keys.end(), name) == supported_keys.end()) {
        THROW_IE_EXCEPTION << "Unsupported config key : " << name;
    }

    Parameter result;
    auto option = _config.find(name);
    if (option != _config.end())
        result = option->second;

    return result;
}

QueryNetworkResult Engine::QueryNetwork(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config) const {
    VPU_PROFILE(QueryNetwork);
    QueryNetworkResult res;

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    const auto deviceName = parsedConfigCopy.deviceName();
    if (!deviceName.empty()) {
        const auto deviceIDs = GetMetric(METRIC_KEY(AVAILABLE_DEVICES), {}).as<std::vector<std::string>>();
        VPU_THROW_UNLESS(!(std::find(deviceIDs.begin(), deviceIDs.end(), deviceName) == deviceIDs.end()), "Myriad device: {} not found.", deviceName);
    }

    if (auto function = network.getFunction()) {
        std::unordered_set<std::string> originalOps;
        for (auto& node : function->get_ops()) {
            originalOps.emplace(node->get_friendly_name());
        }

        auto clonedNetwork = cloneNetwork(network);
        auto convertedNetwork = vpu::FrontEnd::convertNetwork(*clonedNetwork);

        std::unordered_set<std::string> supported;
        std::unordered_set<std::string> unsupported;

        std::unordered_set<std::string> splitNames;
        std::unordered_set<std::string> concatNames;

        ngraph::NodeVector splits;
        ngraph::NodeVector concats;

        const auto isLayerSupported = [this, &splitNames, &concatNames, &concats, &splits](CNNNetworkIterator& layer) -> bool {
                auto node = (*layer)->getNode();
                if (std::dynamic_pointer_cast<const ::ngraph::opset3::Split>(node) != nullptr) {
                    splitNames.emplace(node->get_friendly_name());
                    splits.push_back(node);
                    return false;
                } else if (std::dynamic_pointer_cast<const ::ngraph::opset3::Concat>(node) != nullptr) {
                    concatNames.emplace(node->get_friendly_name());
                    concats.push_back(node);
                    return false;
                } else {
                    auto stageBuilder = std::make_shared<StageBuilder>();
                    auto frontEnd = std::make_shared<FrontEnd>(stageBuilder, GetCore());
                    return frontEnd->isLayerSupported((*layer)->type);
                }
        };

        for (CNNNetworkIterator itLayer{convertedNetwork.get()};
             itLayer != CNNNetworkIterator();
             itLayer++) {
            const auto fusedNode = (*itLayer)->getNode();
            if (fusedNode == nullptr) {
                continue;
            }

            for (auto& fusedLayerName : ngraph::getFusedNamesVector(fusedNode)) {
                if (contains(originalOps, fusedLayerName)) {
                    if (isLayerSupported(itLayer)) {
                        supported.emplace(fusedLayerName);
                    } else {
                        unsupported.emplace(fusedLayerName);
                    }
                }
            }
        }

        for (const auto& layerName : supported) {
            if (contains(unsupported, layerName)) {
                supported.erase(layerName);
            }
        }

        unsupported.clear();

        std::function<void(std::shared_ptr<ngraph::Node>)> markParentSplitAsUnsupported = [&markParentSplitAsUnsupported, &supported, &splitNames]
                                                                                          (const std::shared_ptr<ngraph::Node>& split) {
            const auto inputs = split->inputs();
            for (const auto& input : inputs) {
                const auto& parentName = input.get_source_output().get_node()->get_friendly_name();
                if (contains(supported, parentName) &&
                    contains(splitNames, parentName)) {
                    markParentSplitAsUnsupported(input.get_source_output().get_node_shared_ptr());
                }
            }
            const auto& name = split->get_friendly_name();
            if (contains(supported, name)) {
                supported.erase(name);
            }
        };

        for (const auto& split : splits) {
            // We will mark split as a supported only if all consumers is supported
            bool is_supported = true;
            const auto outputs = split->outputs();
            for (const auto& output : outputs) {
                for (const auto& consumer : output.get_target_inputs()) {
                    const auto& name = consumer.get_node()->get_friendly_name();
                    if (!contains(supported, name) &&
                        !contains(concatNames, name) &&
                        !contains(splitNames, name)) {
                        is_supported = false;
                        break;
                    }
                }
            }
            if (is_supported) {
                supported.emplace(split->get_friendly_name());
            } else {
                // If Split is not supported and it's parent is also Split, mark parent as unsupported
                markParentSplitAsUnsupported(split);
            }
        }

        for (const auto& concat : concats) {
            // We will mark concat as a supported only if all parent layers is supported
            bool is_supported = true;
            const auto inputs = concat->inputs();
            for (const auto& input : inputs) {
                const auto& name = input.get_source_output().get_node()->get_friendly_name();
                if (!contains(supported, name) &&
                    !contains(concatNames, name)) {
                    is_supported = false;
                    break;
                }
            }
            if (is_supported) {
                supported.emplace(concat->get_friendly_name());
            }
        }

        for (const auto& node : function->get_ops()) {
            if (contains(supported, node->get_friendly_name())) {
                for (const auto& inputNodeOutput : node->input_values()) {
                    if (ngraph::op::is_constant(inputNodeOutput.get_node()) || ngraph::op::is_parameter(inputNodeOutput.get_node())) {
                        supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                    }
                }
                for (const auto& outputs : node->outputs()) {
                    for (const auto& outputNodeInput : outputs.get_target_inputs()) {
                        if (ngraph::op::is_output(outputNodeInput.get_node())) {
                            supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                        }
                    }
                }
            }
        }

        for (const auto& layerName : supported) {
            res.supportedLayersMap.emplace(layerName, GetName());
        }
    } else {
        const auto log = std::make_shared<Logger>(
            "GraphCompiler",
            parsedConfigCopy.logLevel(),
            defaultOutput(parsedConfigCopy.compilerLogFilePath()));

        const auto layerNames = getSupportedLayers(
            network,
            static_cast<Platform>(parsedConfigCopy.platform()),
            parsedConfigCopy.compileConfig(),
            log,
            GetCore());

        for (const auto& layerName : layerNames) {
            res.supportedLayersMap.insert({ layerName, GetName() });
        }
    }

    return res;
}

Engine::Engine(std::shared_ptr<IMvnc> mvnc) :
        _mvnc(std::move(mvnc)),
        _metrics(std::make_shared<MyriadMetrics>()) {
    VPU_THROW_UNLESS(_mvnc, "mvnc is null");

    _pluginName = "MYRIAD";

IE_SUPPRESS_DEPRECATED_START
    _config = {
        { MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES) },
        { MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO) },
        { MYRIAD_CUSTOM_LAYERS, "" },
        { MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO) },

        // Deprecated
        { KEY_VPU_HW_STAGES_OPTIMIZATION, CONFIG_VALUE(YES) },
        { KEY_VPU_PRINT_RECEIVE_TENSOR_TIME, CONFIG_VALUE(NO) },
        { KEY_VPU_CUSTOM_LAYERS, "" },
        { KEY_VPU_MYRIAD_FORCE_RESET, CONFIG_VALUE(NO) },
        { KEY_VPU_MYRIAD_PLATFORM, "" },

        { KEY_LOG_LEVEL, CONFIG_VALUE(LOG_NONE) },
        { KEY_EXCLUSIVE_ASYNC_REQUESTS, CONFIG_VALUE(NO) },
        { KEY_PERF_COUNT, CONFIG_VALUE(NO) },
        { KEY_CONFIG_FILE, "" },
        { KEY_DEVICE_ID, "" },
    };
IE_SUPPRESS_DEPRECATED_END
}

InferenceEngine::ExecutableNetwork Engine::ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork =
            std::make_shared<ExecutableNetwork>(
                model, _mvnc, _devicePool, parsedConfigCopy, GetCore());

    return make_executable_network(executableNetwork);
}

InferenceEngine::ExecutableNetwork Engine::ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    std::ifstream blobFile(modelFileName, std::ios::binary);

    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << ie::details::as_status << NETWORK_NOT_READ;
    }

    return ImportNetwork(blobFile, config);
}

InferenceEngine::Parameter Engine::GetMetric(const std::string& name,
                                     const std::map<std::string, InferenceEngine::Parameter> & options) const {
    const auto mvnc = _mvnc;
    const auto metrics = _metrics;
    const auto devicePool = _devicePool;
    const auto getSpecifiedDeviceName = [&mvnc, &metrics, &devicePool, &options]() {
        if (options.count(KEY_DEVICE_ID)) {
            return options.at(KEY_DEVICE_ID).as<std::string>();
        }

        const auto availableDevices = metrics->AvailableDevicesNames(mvnc, devicePool);
        VPU_THROW_UNLESS(!availableDevices.empty(), "No devices available.");
        VPU_THROW_UNLESS(availableDevices.size() == 1, "KEY_DEVICE_ID is undefined.");

        return availableDevices.front();
    };
    const auto getDeviceByName = [&devicePool](const std::string& deviceName) {
        const auto deviceIt = std::find_if(
                devicePool.begin(), devicePool.end(), [&deviceName](DevicePtr device) {
                    return device->_name == deviceName;
                });
        if (deviceIt == devicePool.end()) {
            return DevicePtr();
        }
        return *deviceIt;
    };

    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics->AvailableDevicesNames(_mvnc, _devicePool));
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics->FullName(getSpecifiedDeviceName()));
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        const auto& supportedMetrics = _metrics->SupportedMetrics();
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{supportedMetrics.cbegin(), supportedMetrics.cend()});
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        const auto& supportedConfigKeys = _metrics->SupportedConfigKeys();
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{supportedConfigKeys.cbegin(), supportedConfigKeys.cend()});
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        const auto& optimizationCapabilities = _metrics->OptimizationCapabilities();
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{optimizationCapabilities.cbegin(), optimizationCapabilities.cend()});
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics->RangeForAsyncInferRequests(_config));
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        const auto& device = getDeviceByName(getSpecifiedDeviceName());
        if (device != nullptr) {
            IE_SET_METRIC_RETURN(DEVICE_THERMAL, _metrics->DevicesThermal(device));
        } else {
            return Parameter();
        }
    }
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}
