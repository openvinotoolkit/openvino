// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <tuple>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <cpp/ie_cnn_network.h>
#include <ie_api.h>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <legacy/ie_util_internal.hpp>

#include <vpu/private_plugin_config.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/ngraph/query_network.hpp>

#include <vpu/configuration/options/log_level.hpp>
#include <vpu/configuration/options/copy_optimization.hpp>
#include <vpu/configuration/options/power_config.hpp>
#include <vpu/configuration/options/protocol.hpp>
#include <vpu/configuration/options/hw_acceleration.hpp>
#include <vpu/configuration/options/hw_extra_split.hpp>
#include <vpu/configuration/options/hw_pool_conv_merge.hpp>
#include <vpu/configuration/options/hw_black_list.hpp>
#include <vpu/configuration/options/hw_inject_stages.hpp>
#include <vpu/configuration/options/hw_dilation.hpp>
#include <vpu/configuration/options/tiling_cmx_limit_kb.hpp>
#include <vpu/configuration/options/watchdog_interval.hpp>
#include <vpu/configuration/options/enable_receiving_tensor_time.hpp>
#include <vpu/configuration/options/perf_report_mode.hpp>
#include <vpu/configuration/options/perf_count.hpp>
#include <vpu/configuration/options/pack_data_in_cmx.hpp>
#include <vpu/configuration/options/number_of_shaves.hpp>
#include <vpu/configuration/options/number_of_cmx_slices.hpp>
#include <vpu/configuration/options/throughput_streams.hpp>
#include <vpu/configuration/options/vpu_scales_option.hpp>
#include <vpu/configuration/options/tensor_strides.hpp>
#include <vpu/configuration/options/ignore_unknown_layers.hpp>
#include <vpu/configuration/options/force_pure_tensor_iterator.hpp>
#include <vpu/configuration/options/enable_tensor_iterator_unrolling.hpp>
#include <vpu/configuration/options/exclusive_async_requests.hpp>
#include <vpu/configuration/options/enable_weights_analysis.hpp>
#include <vpu/configuration/options/enable_repl_with_screlu.hpp>
#include <vpu/configuration/options/enable_permute_merging.hpp>
#include <vpu/configuration/options/enable_memory_types_annotation.hpp>
#include <vpu/configuration/options/dump_internal_graph_file_name.hpp>
#include <vpu/configuration/options/dump_all_passes_directory.hpp>
#include <vpu/configuration/options/dump_all_passes.hpp>
#include <vpu/configuration/options/device_id.hpp>
#include <vpu/configuration/options/device_connect_timeout.hpp>
#include <vpu/configuration/options/disable_convert_stages.hpp>
#include <vpu/configuration/options/disable_reorder.hpp>
#include <vpu/configuration/options/detect_network_batch.hpp>
#include <vpu/configuration/options/custom_layers.hpp>
#include <vpu/configuration/options/config_file.hpp>
#include <vpu/configuration/options/memory_type.hpp>
#include <vpu/configuration/options/enable_force_reset.hpp>
#include <vpu/configuration/options/check_preprocessing_inside_model.hpp>
#include <vpu/configuration/options/enable_early_eltwise_relu_fusion.hpp>
#include <vpu/configuration/options/enable_custom_reshape_param.hpp>
#include <vpu/configuration/options/none_layers.hpp>
#include <vpu/configuration/options/enable_async_dma.hpp>
#include <vpu/configuration/options/enable_mx_boot.hpp>
#include "vpu/configuration/options/performance_hint.hpp"
#include "vpu/configuration/options/performance_hint_num_requests.hpp"
#include "vpu/configuration/options/ov_throughput_streams.hpp"

#include "myriad_plugin.h"

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_myriad/myriad_properties.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace vpu::MyriadPlugin;


IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(LoadExeNetworkImpl);

    auto executableNetworkConfiguration = _parsedConfig;
    executableNetworkConfiguration.from(config);
    executableNetworkConfiguration.validate();

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(network, _mvnc, _devicePool, executableNetworkConfiguration, GetCore());
    executableNetwork->SetPointerToPlugin(shared_from_this());
    return executableNetwork;
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    _parsedConfig.from(config);

    // TODO: remove once all options are migrated
    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }

#ifndef NDEBUG
    if (const auto envVar = std::getenv("IE_VPU_LOG_LEVEL")) {
        _parsedConfig.set(LogLevelOption::key(), envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_TILING_CMX_LIMIT_KB")) {
        _parsedConfig.set(TilingCMXLimitKBOption::key(), envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_MYRIAD_WATCHDOG_INTERVAL")) {
        _parsedConfig.set(WatchdogIntervalOption::key(), envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_NUMBER_OF_SHAVES_AND_CMX_SLICES")) {
        _parsedConfig.set(NumberOfSHAVEsOption::key(), envVar);
        _parsedConfig.set(NumberOfCMXSlicesOption::key(), envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_FILE_NAME")) {
        _parsedConfig.set(DumpInternalGraphFileNameOption::key(), envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_DIRECTORY")) {
        _parsedConfig.set(DumpAllPassesDirectoryOption::key(), envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_ALL_PASSES")) {
        _parsedConfig.set(DumpAllPassesOption::key(), std::stoi(envVar) != 0
            ? InferenceEngine::PluginConfigParams::YES : InferenceEngine::PluginConfigParams::NO);
    }
    if (const auto envVar = std::getenv("IE_VPU_MYRIAD_FORCE_RESET")) {
        _parsedConfig.set(EnableForceResetOption::key(), std::stoi(envVar) != 0
            ? InferenceEngine::PluginConfigParams::YES : InferenceEngine::PluginConfigParams::NO);
    }
#endif
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    // TODO: remove once all options are migrated
    const auto& supportedKeys = _metrics->SupportedConfigKeys();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(supportedKeys.count(name) == 1 || _parsedConfig.supports(name), "Unsupported configuration key: {}", name);

    Parameter result;
    if (_parsedConfig.supports(name)) {
        result = _parsedConfig.asParameter(name);
    } else if (_config.count(name)) {
        // TODO: remove once all options are migrated
        result = _config.at(name);
    }

    return result;
}

QueryNetworkResult Engine::QueryNetwork(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config) const {
    VPU_PROFILE(QueryNetwork);
    QueryNetworkResult res;

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.from(config);

    const auto deviceName = parsedConfigCopy.get<DeviceIDOption>();
    if (!deviceName.empty()) {
        const auto deviceIDs = GetMetric(METRIC_KEY(AVAILABLE_DEVICES), {}).as<std::vector<std::string>>();
        VPU_THROW_UNLESS(!(std::find(deviceIDs.begin(), deviceIDs.end(), deviceName) == deviceIDs.end()), "Myriad device: {} not found.", deviceName);
    }

    const auto log = std::make_shared<Logger>(
            "GraphCompiler",
            _parsedConfig.get<LogLevelOption>(),
            consoleOutput());
    std::set<std::string> namesToExclude;
    const auto supportedNetworks = vpu::FrontEnd::checkSupportedNetworks(network, namesToExclude);
    for (const auto& supportedNetwork : supportedNetworks) {
        const auto supportedLayers = getSupportedLayers(
                supportedNetwork,
                parsedConfigCopy,
                log,
                GetCore(),
                namesToExclude);

        if (auto function = supportedNetwork.getFunction()) {
            auto clonedNetwork = cloneNetwork(supportedNetwork);
            auto clonedFunction = clonedNetwork.getFunction();
            auto convertedNetwork = vpu::FrontEnd::convertNetwork(clonedNetwork);

            QueryNetworkResult supportedRes = getQueryNetwork(clonedNetwork, function, GetName(), supportedLayers);
            auto removedNodeNames = GetRemovedNodes(function, clonedFunction);

            for (const auto& layer : removedNodeNames) {
                res.supportedLayersMap.emplace(layer, GetName());
            }

            for (const auto& layer : supportedRes.supportedLayersMap) {
                res.supportedLayersMap.insert(layer);
            }
        } else {
            for (const auto& layerName : supportedLayers) {
                res.supportedLayersMap.insert({ layerName, GetName() });
            }
        }
    }

    return res;
}

Engine::Engine(std::shared_ptr<IMvnc> mvnc) :
        _mvnc(std::move(mvnc)),
        _metrics(std::make_shared<MyriadMetrics>()) {
    VPU_THROW_UNLESS(_mvnc, "mvnc is null");

    _pluginName = "MYRIAD";

    _parsedConfig.registerOption<LogLevelOption>();
    _parsedConfig.registerOption<CopyOptimizationOption>();
    _parsedConfig.registerOption<PowerConfigOption>();
    _parsedConfig.registerOption<ProtocolOption>();
    _parsedConfig.registerOption<HwAccelerationOption>();
    _parsedConfig.registerOption<HwExtraSplitOption>();
    _parsedConfig.registerOption<HwPoolConvMergeOption>();
    _parsedConfig.registerOption<HwBlackListOption>();
    _parsedConfig.registerOption<HwInjectStagesOption>();
    _parsedConfig.registerOption<HwDilationOption>();
    _parsedConfig.registerOption<TilingCMXLimitKBOption>();
    _parsedConfig.registerOption<WatchdogIntervalOption>();
    _parsedConfig.registerOption<EnableReceivingTensorTimeOption>();
    _parsedConfig.registerOption<PerfReportModeOption>();
    _parsedConfig.registerOption<PerfCountOption>();
    _parsedConfig.registerOption<PackDataInCMXOption>();
    _parsedConfig.registerOption<NumberOfSHAVEsOption>();
    _parsedConfig.registerOption<NumberOfCMXSlicesOption>();
    _parsedConfig.registerOption<ThroughputStreamsOption>();
    _parsedConfig.registerOption<VPUScalesOption>();
    _parsedConfig.registerOption<TensorStridesOption>();
    _parsedConfig.registerOption<IgnoreUnknownLayersOption>();
    _parsedConfig.registerOption<ForcePureTensorIteratorOption>();
    _parsedConfig.registerOption<EnableTensorIteratorUnrollingOption>();
    _parsedConfig.registerOption<ExclusiveAsyncRequestsOption>();
    _parsedConfig.registerOption<EnableWeightsAnalysisOption>();
    _parsedConfig.registerOption<EnableReplWithSCReluOption>();
    _parsedConfig.registerOption<EnablePermuteMergingOption>();
    _parsedConfig.registerOption<EnableMemoryTypesAnnotationOption>();
    _parsedConfig.registerOption<DumpInternalGraphFileNameOption>();
    _parsedConfig.registerOption<DumpAllPassesDirectoryOption>();
    _parsedConfig.registerOption<DumpAllPassesOption>();
    _parsedConfig.registerOption<DeviceIDOption>();
    _parsedConfig.registerOption<DeviceConnectTimeoutOption>();
    _parsedConfig.registerOption<DetectNetworkBatchOption>();
    _parsedConfig.registerOption<CustomLayersOption>();
    _parsedConfig.registerOption<ConfigFileOption>();
    _parsedConfig.registerOption<MemoryTypeOption>();
    _parsedConfig.registerOption<EnableForceResetOption>();
    _parsedConfig.registerOption<CheckPreprocessingInsideModelOption>();
    _parsedConfig.registerOption<EnableEarlyEltwiseReluFusionOption>();
    _parsedConfig.registerOption<EnableCustomReshapeParamOption>();
    _parsedConfig.registerOption<NoneLayersOption>();
    _parsedConfig.registerOption<EnableAsyncDMAOption>();
    _parsedConfig.registerOption<EnableMXBootOption>();
    _parsedConfig.registerOption<PerformanceHintOption>();
    _parsedConfig.registerOption<PerformanceHintNumRequestsOption>();
    _parsedConfig.registerOption<OvThroughputStreamsOption>();
IE_SUPPRESS_DEPRECATED_START
    _parsedConfig.registerDeprecatedOption<DisableConvertStagesOption>(InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES);
    _parsedConfig.registerDeprecatedOption<DisableReorderOption>(InferenceEngine::MYRIAD_DISABLE_REORDER);
IE_SUPPRESS_DEPRECATED_END
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    auto executableNetworkConfiguration = _parsedConfig;
    executableNetworkConfiguration.fromAtRuntime(config);
    executableNetworkConfiguration.validate();

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(model, _mvnc, _devicePool, executableNetworkConfiguration, GetCore());
    executableNetwork->SetPointerToPlugin(shared_from_this());
    return executableNetwork;
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
        const auto deviceIt = std::find_if(devicePool.begin(), devicePool.end(), [&deviceName](DevicePtr device) {
            return device->_name == deviceName;
        });
        if (deviceIt == devicePool.end()) {
            return DevicePtr();
        }
        return *deviceIt;
    };

    if (ov::available_devices == name) {
        return _metrics->AvailableDevicesNames(_mvnc, _devicePool);
    } else if (ov::device::full_name == name) {
        return _metrics->FullName(getSpecifiedDeviceName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        const auto& supportedMetrics = _metrics->SupportedMetrics();
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             std::vector<std::string>{supportedMetrics.cbegin(), supportedMetrics.cend()});
    } else if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type  {
            ov::available_devices.name(),
            ov::device::full_name.name(),
            ov::supported_properties.name(),
            ov::device::capabilities.name(),
            ov::range_for_async_infer_requests.name(),
            ov::device::thermal.name(),
            ov::device::architecture.name(),
            ov::num_streams.name(),
            ov::hint::performance_mode.name(),
            ov::hint::num_requests.name(),
        };
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        // TODO: remove once all options are migrated
        auto supportedConfigKeys = _metrics->SupportedConfigKeys();
        const auto& publicKeys = _parsedConfig.getPublicKeys();
        supportedConfigKeys.insert(publicKeys.cbegin(), publicKeys.cend());
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS,
                             std::vector<std::string>{supportedConfigKeys.cbegin(), supportedConfigKeys.cend()});
    } else if (ov::device::capabilities == name) {
        return std::vector<std::string> {_metrics->OptimizationCapabilities().begin(), _metrics->OptimizationCapabilities().end()};
    } else if (ov::range_for_async_infer_requests == name) {
        return _metrics->RangeForAsyncInferRequests(_config);
    } else if (ov::device::architecture == name) {
        return _metrics->DeviceArchitecture(options);
    } else if (ov::device::thermal == name) {
        const auto& device = getDeviceByName(getSpecifiedDeviceName());
        if (device != nullptr) {
            return _metrics->DevicesThermal(device);
        } else {
            return Parameter();
        }
    } else if (name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    }
    IE_THROW(NotImplemented);
}
