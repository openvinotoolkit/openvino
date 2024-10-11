// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cid_compiler_adapter.hpp"

#include <ze_graph_ext.h>

#include "cid_graph.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "ze_intel_npu_uuid.h"
#include "zero_backend.hpp"
#include "zero_device.hpp"
#include "zero_init.hpp"

namespace intel_npu {

CidCompilerAdapter::CidCompilerAdapter(const std::shared_ptr<IEngineBackend>& iEngineBackend)
    : _logger("CidCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize CidCompilerAdapter start");

    std::shared_ptr<ZeroEngineBackend> zeroBackend = nullptr;
    zeroBackend = std::dynamic_pointer_cast<ZeroEngineBackend>(iEngineBackend);
    if (!zeroBackend) {
        OPENVINO_THROW("CidCompilerAdapter init failed to cast zeroBackend, zeroBackend is a nullptr");
    }

    ze_context_handle_t zeContext = static_cast<ze_context_handle_t>(zeroBackend->getContext());
    ze_driver_handle_t driverHandle = static_cast<ze_driver_handle_t>(zeroBackend->getDriverHandle());
    ze_device_handle_t deviceHandle = static_cast<ze_device_handle_t>(zeroBackend->getDeviceHandle());
    ze_graph_dditable_ext_curr_t& graphDdiTableExt = zeroBackend->getGraphDdiTable();
    ze_command_queue_npu_dditable_ext_curr_t& commandQueueDdiTable = zeroBackend->getCommandQueueDdiTable();

    uint32_t graphExtVersion = graphDdiTableExt.version();

    std::shared_ptr<ZeroDevice> zeroDevice = nullptr;
    zeroDevice = std::dynamic_pointer_cast<ZeroDevice>(zeroBackend->getDevice());
    if (!zeroDevice) {
        OPENVINO_THROW("CidCompilerAdapter init failed to cast zeroDevice, zeroDevice is a nullptr");
    }
    auto group_ordinal = zeroDevice->getGroupOrdinal();

    if (driverHandle == nullptr) {
        OPENVINO_THROW("CidCompilerAdapter failed to get properties about zeDriver");
    }

    _logger.info("CidCompilerAdapter creating adapter using graphExtVersion");

    switch (graphExtVersion) {
    case ZE_GRAPH_EXT_VERSION_1_3:
        _apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_3_t>>(driverHandle,
                                                                                               deviceHandle,
                                                                                               zeContext,
                                                                                               graphDdiTableExt,
                                                                                               commandQueueDdiTable,
                                                                                               group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_4:
        _apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>>(driverHandle,
                                                                                               deviceHandle,
                                                                                               zeContext,
                                                                                               graphDdiTableExt,
                                                                                               commandQueueDdiTable,
                                                                                               group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_5:
        _apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_5_t>>(driverHandle,
                                                                                               deviceHandle,
                                                                                               zeContext,
                                                                                               graphDdiTableExt,
                                                                                               commandQueueDdiTable,
                                                                                               group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_6:
        _apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_6_t>>(driverHandle,
                                                                                               deviceHandle,
                                                                                               zeContext,
                                                                                               graphDdiTableExt,
                                                                                               commandQueueDdiTable,
                                                                                               group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_7:
        _apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_7_t>>(driverHandle,
                                                                                               deviceHandle,
                                                                                               zeContext,
                                                                                               graphDdiTableExt,
                                                                                               commandQueueDdiTable,
                                                                                               group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_8:
        _apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_8_t>>(driverHandle,
                                                                                               deviceHandle,
                                                                                               zeContext,
                                                                                               graphDdiTableExt,
                                                                                               commandQueueDdiTable,
                                                                                               group_ordinal);
        break;
    default:
        _apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_2_t>>(driverHandle,
                                                                                               deviceHandle,
                                                                                               zeContext,
                                                                                               graphDdiTableExt,
                                                                                               commandQueueDdiTable,
                                                                                               group_ordinal);
        break;
    }

    _logger.info("initialize CidCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> CidCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                    const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "CidCompilerAdapter", "compile");

    _logger.debug("compile start");
    ze_graph_handle_t graphHandle = _apiAdapter->compile(model, config);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _apiAdapter->getNetworkMeta(graphHandle);
    networkMeta.name = model->get_friendly_name();

    return std::make_shared<CidGraph>(_apiAdapter, graphHandle, std::move(networkMeta), config);
}

std::shared_ptr<IGraph> CidCompilerAdapter::parse(const std::vector<uint8_t>& network, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "CidCompilerAdapter", "parse");

    _logger.debug("parse start");
    ze_graph_handle_t graphHandle = _apiAdapter->parse(network, config);
    _logger.debug("parse end");

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    const auto networkMeta = _apiAdapter->getNetworkMeta(graphHandle);

    return std::make_shared<CidGraph>(_apiAdapter, graphHandle, std::move(networkMeta), config);
}

ov::SupportedOpsMap CidCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "CidCompilerAdapter", "query");

    return _apiAdapter->query(model, config);
}

}  // namespace intel_npu
