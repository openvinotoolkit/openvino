// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <tuple>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <legacy/ie_util_internal.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>
#include <vpu/vpu_plugin_config.hpp>
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
#include <vpu/configuration/options/ir_with_scales_directory.hpp>
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
#include <vpu/configuration/options/disable_convert_stages.hpp>
#include <vpu/configuration/options/disable_reorder.hpp>
#include <vpu/configuration/options/device_id.hpp>
#include <vpu/configuration/options/device_connect_timeout.hpp>
#include <vpu/configuration/options/detect_network_batch.hpp>
#include <vpu/configuration/options/custom_layers.hpp>
#include <vpu/configuration/options/config_file.hpp>
#include <vpu/configuration/options/memory_type.hpp>
#include <vpu/configuration/options/enable_force_reset.hpp>
#include <vpu/configuration/options/platform.hpp>
#include <vpu/configuration/options/check_preprocessing_inside_model.hpp>
#include <vpu/configuration/options/enable_early_eltwise_relu_fusion.hpp>
#include <vpu/configuration/options/enable_custom_reshape_param.hpp>
#include <vpu/configuration/options/none_layers.hpp>
#include <vpu/configuration/options/enable_async_dma.hpp>

#include "myriad_plugin.h"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::VPUConfigParams;
using namespace vpu::MyriadPlugin;


IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(LoadExeNetworkImpl);

    auto executableNetworkConfiguration = _parsedConfig;
    executableNetworkConfiguration.from(config);
    executableNetworkConfiguration.validate();

    return std::make_shared<ExecutableNetwork>(network, _mvnc, _devicePool, executableNetworkConfiguration, GetCore());
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

    const auto supportedLayers = getSupportedLayers(
            network,
            ncDevicePlatform_t::NC_ANY_PLATFORM,
            parsedConfigCopy,
            log,
            GetCore());

    if (auto function = network.getFunction()) {
        auto clonedNetwork = cloneNetwork(network);
        auto convertedNetwork = vpu::FrontEnd::convertNetwork(clonedNetwork);

        res = getQueryNetwork(convertedNetwork, function, GetName(), supportedLayers);
    } else {
        for (const auto& layerName : supportedLayers) {
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
    _parsedConfig.registerOption<IRWithScalesDirectoryOption>();
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

IE_SUPPRESS_DEPRECATED_START
    _parsedConfig.registerDeprecatedOption<DisableConvertStagesOption>(InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES);
    _parsedConfig.registerDeprecatedOption<DisableReorderOption>(InferenceEngine::MYRIAD_DISABLE_REORDER);
    _parsedConfig.registerDeprecatedOption<LogLevelOption>(VPU_CONFIG_KEY(LOG_LEVEL));
    _parsedConfig.registerDeprecatedOption<ProtocolOption>(VPU_MYRIAD_CONFIG_KEY(PROTOCOL));
    _parsedConfig.registerDeprecatedOption<HwAccelerationOption>(VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION));
    _parsedConfig.registerDeprecatedOption<EnableReceivingTensorTimeOption>(VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME));
    _parsedConfig.registerDeprecatedOption<DetectNetworkBatchOption>(VPU_CONFIG_KEY(DETECT_NETWORK_BATCH));
    _parsedConfig.registerDeprecatedOption<CustomLayersOption>(VPU_CONFIG_KEY(CUSTOM_LAYERS));
    _parsedConfig.registerDeprecatedOption<MemoryTypeOption>(VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE));
    _parsedConfig.registerDeprecatedOption<EnableForceResetOption>(VPU_MYRIAD_CONFIG_KEY(FORCE_RESET));
    _parsedConfig.registerDeprecatedOption<PlatformOption>(VPU_MYRIAD_CONFIG_KEY(PLATFORM));
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
        // TODO: remove once all options are migrated
        auto supportedConfigKeys = _metrics->SupportedConfigKeys();
        const auto& publicKeys = _parsedConfig.getPublicKeys();
        supportedConfigKeys.insert(publicKeys.cbegin(), publicKeys.cend());
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{supportedConfigKeys.cbegin(), supportedConfigKeys.cend()});
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        const auto& optimizationCapabilities = _metrics->OptimizationCapabilities();
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{optimizationCapabilities.cbegin(), optimizationCapabilities.cend()});
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics->RangeForAsyncInferRequests(_config));
    } else if (name == METRIC_KEY(DEVICE_ARCHITECTURE)) {
        IE_SET_METRIC_RETURN(DEVICE_ARCHITECTURE, _metrics->DeviceArchitecture(options));
    } else if (name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        const auto& device = getDeviceByName(getSpecifiedDeviceName());
        if (device != nullptr) {
            IE_SET_METRIC_RETURN(DEVICE_THERMAL, _metrics->DevicesThermal(device));
        } else {
            return Parameter();
        }
    }
    IE_THROW(NotImplemented);
}
