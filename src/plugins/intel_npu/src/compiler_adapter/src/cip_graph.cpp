// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cip_graph.hpp"

#include "intel_npu/utils/zero/zero_result.hpp"
#include "zero_backend.hpp"
#include "zero_device.hpp"
#include "zero_init.hpp"
#include "zero_wrappers.hpp"

namespace intel_npu {

CipGraph::CipGraph(const std::shared_ptr<IEngineBackend>& iEngineBackend,
                   NetworkMetadata metadata,
                   std::vector<uint8_t> compiledNetwork,
                   const Config& config)
    : IGraph(nullptr, std::move(metadata)),
      _zeroBackend(std::dynamic_pointer_cast<ZeroEngineBackend>(iEngineBackend)),
      _compiledNetwork(std::move(compiledNetwork)),
      _config(config),
      _logger("CidGraph", _config.get<LOG_LEVEL>()) {
    if (_zeroBackend == nullptr) {
        return;
    }

    auto zeroDevice = std::dynamic_pointer_cast<ZeroDevice>(_zeroBackend->getDevice());
    if (zeroDevice == nullptr) {
        return;
    }

    auto zeContext = static_cast<ze_context_handle_t>(_zeroBackend->getContext());
    auto deviceHandle = static_cast<ze_device_handle_t>(_zeroBackend->getDeviceHandle());
    auto& graphDdiTableExt = _zeroBackend->getGraphDdiTable();

    ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                         nullptr,
                         ZE_GRAPH_FORMAT_NATIVE,
                         compiledNetwork.size(),
                         compiledNetwork.data(),
                         nullptr};

    ze_graph_handle_t graphHandle = nullptr;
    auto result = graphDdiTableExt.pfnCreate(zeContext, deviceHandle, &desc, &graphHandle);
    OPENVINO_ASSERT(result == ZE_RESULT_SUCCESS,
                    "Failed to set argument value",
                    " result: ",
                    ze_result_to_string(result),
                    ", code 0x",
                    std::hex,
                    uint64_t(result));

    // update graph handle
    _handle = static_cast<void*>(graphHandle);

    initialize();
}

CompiledNetwork CipGraph::export_blob() const {
    return CompiledNetwork(_compiledNetwork.data(), _compiledNetwork.size(), _compiledNetwork);
}

std::vector<ov::ProfilingInfo> CipGraph::process_profiling_output() const {
    OPENVINO_THROW("Profiling post-processing is not implemented.");
}

void CipGraph::set_argument_value(uint32_t argi, const void* argv) const {
    auto& graphDdiTableExt = _zeroBackend->getGraphDdiTable();

    auto result = graphDdiTableExt.pfnSetArgumentValue(static_cast<ze_graph_handle_t>(_handle), argi, argv);

    OPENVINO_ASSERT(result == ZE_RESULT_SUCCESS,
                    "Failed to set argument value",
                    " result: ",
                    ze_result_to_string(result),
                    ", code 0x",
                    std::hex,
                    uint64_t(result));
}

void CipGraph::initialize() const {
    auto& graphDdiTableExt = _zeroBackend->getGraphDdiTable();

    if (graphDdiTableExt.version() < ZE_GRAPH_EXT_VERSION_1_8) {
        initialize_graph_through_command_list();
    } else {
        ze_graph_properties_2_t properties = {};
        properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        graphDdiTableExt.pfnGetProperties2(static_cast<ze_graph_handle_t>(_handle), &properties);

        if (properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
            graphDdiTableExt.pfnGraphInitialize(static_cast<ze_graph_handle_t>(_handle));
        }

        if (properties.initStageRequired & ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) {
            initialize_graph_through_command_list();
        }
    }
}

void CipGraph::initialize_graph_through_command_list() const {
    auto zeContext = static_cast<ze_context_handle_t>(_zeroBackend->getContext());
    auto deviceHandle = static_cast<ze_device_handle_t>(_zeroBackend->getDeviceHandle());

    auto& graphDdiTableExt = _zeroBackend->getGraphDdiTable();
    auto& commandQueueDdiTableExt = _zeroBackend->getCommandQueueDdiTable();

    auto zeroDevice = std::dynamic_pointer_cast<ZeroDevice>(_zeroBackend->getDevice());
    if (!zeroDevice) {
        OPENVINO_THROW("CidCompilerAdapter init failed to cast zeroDevice, zeroDevice is a nullptr");
    }
    auto group_ordinal = zeroDevice->getGroupOrdinal();

    _logger.debug("ZeroExecutor::ZeroExecutor init start - create graph_command_list");
    CommandList graph_command_list(deviceHandle, zeContext, graphDdiTableExt, _config, group_ordinal);
    _logger.debug("ZeroExecutor::ZeroExecutor - create graph_command_queue");
    CommandQueue graph_command_queue(deviceHandle,
                                     zeContext,
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                     commandQueueDdiTableExt,
                                     _config,
                                     group_ordinal);
    _logger.debug("ZeroExecutor::ZeroExecutor - create fence");
    Fence fence(graph_command_queue, _config);

    _logger.debug("ZeroExecutor::ZeroExecutor - performing appendGraphInitialize");
    graph_command_list.appendGraphInitialize(static_cast<ze_graph_handle_t>(_handle));
    _logger.debug("ZeroExecutor::ZeroExecutor - closing graph command list");
    graph_command_list.close();

    _logger.debug("ZeroExecutor::ZeroExecutor - performing executeCommandList");
    graph_command_queue.executeCommandList(graph_command_list, fence);
    _logger.debug("ZeroExecutor::ZeroExecutor - performing hostSynchronize");
    fence.hostSynchronize();
    _logger.debug("ZeroExecutor::ZeroExecutor - hostSynchronize completed");
}

CipGraph::~CipGraph() {
    if (_handle != nullptr) {
        auto& graphDdiTableExt = _zeroBackend->getGraphDdiTable();

        auto result = graphDdiTableExt.pfnDestroy(static_cast<ze_graph_handle_t>(_handle));
        if (ZE_RESULT_SUCCESS != result) {
            _logger.error("graphDdiTableExt.pfnDestroy failed %#X", uint64_t(result));
        }
    }
}

}  // namespace intel_npu
