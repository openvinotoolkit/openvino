// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <ze_graph_ext.h>
#include <ze_intel_npu_uuid.h>

#include "cid_graph.hpp"
#include "driver_compiler_utils.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

DriverCompilerAdapter::DriverCompilerAdapter(const std::shared_ptr<IDevice>& device)
    : _adapter(device->createAdapter()),
      _logger("DriverCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize DriverCompilerAdapter start");
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compile");

    const ze_graph_compiler_version_info_t& compilerVersion = _adapter->getDeviceGraphProperties().compilerVersion;
    const auto maxOpsetVersion = _adapter->getDeviceGraphProperties().maxOVOpsetVersionSupported;
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
    ze_graph_handle_t graphHandle = _adapter->getGraphHandle(std::move(serializedIR), buildFlags, flags);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _adapter->getNetworkMeta(graphHandle);
    networkMeta.name = model->get_friendly_name();

    return std::make_shared<CidGraph>(_adapter, graphHandle, std::move(networkMeta), config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::parse(const std::vector<uint8_t>& network, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "parse");

    _logger.debug("parse start");
    ze_graph_handle_t graphHandle = _adapter->getGraphHandle(network);
    _logger.debug("parse end");

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    auto networkMeta = _adapter->getNetworkMeta(graphHandle);

    return std::make_shared<CidGraph>(_adapter, graphHandle, std::move(networkMeta), config);
}

ov::SupportedOpsMap DriverCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "query");

    const ze_graph_compiler_version_info_t& compilerVersion = _adapter->getDeviceGraphProperties().compilerVersion;
    const auto maxOpsetVersion = _adapter->getDeviceGraphProperties().maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    std::string buildFlags;
    buildFlags += serializeConfig(config, compilerVersion);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    ov::SupportedOpsMap result;
    const std::string deviceName = "NPU";

    try {
        const auto supportedLayers = _adapter->queryResultFromSupportedLayers(std::move(serializedIR), buildFlags);
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
