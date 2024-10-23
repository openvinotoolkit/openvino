// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <ze_graph_ext.h>

#include "cid_graph.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "ze_intel_npu_uuid.h"
#include "zero_device.hpp"
#include "zero_init.hpp"
#include "zero_link.hpp"

namespace intel_npu {

DriverCompilerAdapter::DriverCompilerAdapter(const std::shared_ptr<IEngineBackend>& iEngineBackend)
    : _logger("DriverCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize DriverCompilerAdapter start");

    _zeroBackend = std::dynamic_pointer_cast<ZeroEngineBackend>(iEngineBackend);
    if (!_zeroBackend) {
        OPENVINO_THROW("DriverCompilerAdapter init failed to cast zeroBackend, zeroBackend is a nullptr");
    }

    ze_context_handle_t zeContext = static_cast<ze_context_handle_t>(_zeroBackend->getContext());
    ze_driver_handle_t driverHandle = static_cast<ze_driver_handle_t>(_zeroBackend->getDriverHandle());
    ze_device_handle_t deviceHandle = static_cast<ze_device_handle_t>(_zeroBackend->getDeviceHandle());
    ze_graph_dditable_ext_curr_t& graphDdiTableExt = _zeroBackend->getGraphDdiTable();
    ze_command_queue_npu_dditable_ext_curr_t& commandQueueDdiTable = _zeroBackend->getCommandQueueDdiTable();

    uint32_t graphExtVersion = graphDdiTableExt.version();

    _deviceGraphProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    auto result = graphDdiTableExt.pfnDeviceGetGraphProperties(deviceHandle, &_deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to compile network. L0 pfnDeviceGetGraphProperties",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    std::shared_ptr<ZeroDevice> zeroDevice = nullptr;
    zeroDevice = std::dynamic_pointer_cast<ZeroDevice>(_zeroBackend->getDevice());
    if (!zeroDevice) {
        OPENVINO_THROW("DriverCompilerAdapter init failed to cast zeroDevice, zeroDevice is a nullptr");
    }
    auto group_ordinal = zeroDevice->getGroupOrdinal();

    if (driverHandle == nullptr) {
        OPENVINO_THROW("DriverCompilerAdapter failed to get properties about Driver");
    }

    _logger.info("DriverCompilerAdapter creating adapter using graphExtVersion");

    switch (graphExtVersion) {
    case ZE_GRAPH_EXT_VERSION_1_3:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_3_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_4:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_4_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_5:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_5_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_6:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_6_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_7:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_7_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_8:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_8_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    default:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_2_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    }

    _logger.info("initialize DriverCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compile");

    const ze_graph_compiler_version_info_t& compilerVersion = _deviceGraphProperties.compilerVersion;
    const auto maxOpsetVersion = _deviceGraphProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    _logger.debug("build flags");
    buildFlags += serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += serializeConfig(config, compilerVersion);

    _logger.debug("compileIR Build flags : %s", buildFlags.c_str());

    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = config.get<CACHE_DIR>();
    if (!set_cache_dir.empty() || config.get<BYPASS_UMD_CACHING>()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

    _logger.debug("compile start");
    ze_graph_handle_t graphHandle = _zeroLink->getGraphHandle(std::move(serializedIR), buildFlags, flags);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeroLink->getNetworkMeta(graphHandle);
    networkMeta.name = model->get_friendly_name();

    return std::make_shared<CidGraph>(_zeroLink, graphHandle, std::move(networkMeta), config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::parse(const std::vector<uint8_t>& network, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "parse");

    _logger.debug("parse start");
    ze_graph_handle_t graphHandle = _zeroLink->getGraphHandle(network);
    _logger.debug("parse end");

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeroLink->getNetworkMeta(graphHandle);

    return std::make_shared<CidGraph>(_zeroLink, graphHandle, std::move(networkMeta), config);
}

ov::SupportedOpsMap DriverCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "query");

    const ze_graph_compiler_version_info_t& compilerVersion = _deviceGraphProperties.compilerVersion;
    const auto maxOpsetVersion = _deviceGraphProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    std::string buildFlags;
    buildFlags += serializeConfig(config, compilerVersion);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    ov::SupportedOpsMap result;
    const std::string deviceName = "NPU";

    try {
        const auto supportedLayers = _zeroLink->queryResultFromSupportedLayers(std::move(serializedIR), buildFlags);
        for (auto&& layerName : supportedLayers) {
            result.emplace(layerName, deviceName);
        }
        _logger.info("For given model, there are %d supported layers", supportedLayers.size());
    } catch (std::exception& e) {
        OPENVINO_THROW("Fail in calling querynetwork : ", e.what());
    }

    _logger.debug("query end");
    return result;
}

}  // namespace intel_npu
