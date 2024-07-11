// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_executor.hpp"

#include <ze_api.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/al/prefix.hpp"
#include "openvino/runtime/properties.hpp"
#include "ze_command_queue_npu_ext.h"
#include "zero_device.hpp"
#include "zero_utils.hpp"

using namespace intel_npu;

ZeroExecutor::ZeroExecutor(const std::shared_ptr<const ZeroInitStructsHolder>& initStructs,
                           const std::shared_ptr<const NetworkDescription>& networkDescription,
                           const Config& config,
                           const uint32_t& group_ordinal)
    : _config(config),
      _logger("Graph", _config.get<LOG_LEVEL>()),
      _initStructs(initStructs),
      _networkDesc(networkDescription),
      _graph_ddi_table_ext(_initStructs->getGraphDdiTable()),
      _group_ordinal(group_ordinal),
      _command_queues{{std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                                      _initStructs->getContext(),
                                                      zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                      _initStructs->getCommandQueueDdiTable(),
                                                      _config,
                                                      group_ordinal),
                       std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                                      _initStructs->getContext(),
                                                      zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                      _initStructs->getCommandQueueDdiTable(),
                                                      _config,
                                                      group_ordinal),
                       std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                                      _initStructs->getContext(),
                                                      zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                      _initStructs->getCommandQueueDdiTable(),
                                                      _config,
                                                      group_ordinal)}} {
    _logger.trace("ZeroExecutor::ZeroExecutor init start - create graph_command_list");
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::ZeroExecutor");
    CommandList graph_command_list(_initStructs->getDevice(),
                                   _initStructs->getContext(),
                                   _initStructs->getGraphDdiTable(),
                                   _config,
                                   _group_ordinal);
    _logger.trace("ZeroExecutor::ZeroExecutor - create graph_command_queue");
    CommandQueue graph_command_queue(_initStructs->getDevice(),
                                     _initStructs->getContext(),
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                     _initStructs->getCommandQueueDdiTable(),
                                     _config,
                                     _group_ordinal);
    _logger.trace("ZeroExecutor::ZeroExecutor - create fence");
    Fence fence(graph_command_queue, _config);

    _logger.trace("ZeroExecutor::ZeroExecutor - create graph");
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_GRAPH, itt::domains::LevelZeroBackend, "Executor::ZeroExecutor", "graphCreate");

    ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                         nullptr,
                         ZE_GRAPH_FORMAT_NATIVE,
                         _networkDesc->compiledNetwork.size(),
                         _networkDesc->compiledNetwork.data(),
                         nullptr};
    zeroUtils::throwOnFail(
        "pfnCreate",
        _graph_ddi_table_ext->pfnCreate(_initStructs->getContext(), _initStructs->getDevice(), &desc, &_graph));

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetProperties");
    zeroUtils::throwOnFail("pfnGetProperties", _graph_ddi_table_ext->pfnGetProperties(_graph, &_props));

    auto targetDriverExtVersion = _initStructs->getDriverExtVersion();
    if (targetDriverExtVersion <= ZE_GRAPH_EXT_VERSION_1_1) {
        OPENVINO_THROW("Incompatibility between the NPU plugin and driver! The driver version is too old, please "
                       "update the driver version");
    }

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetArgumentProperties3");
    _logger.trace("ZeroExecutor::ZeroExecutor - performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3;
        zeroUtils::throwOnFail("pfnGetArgumentProperties3",
                               _graph_ddi_table_ext->pfnGetArgumentProperties3(_graph, index, &arg3));

        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg3.type) {
            if (isStateInputName(arg3.name) || isShapeTensorName(arg3.name)) {
                _inputs_desc_map.emplace(std::make_pair(std::string(arg3.name), ArgumentDescriptor{arg3, index}));

            } else {
                _inputs_desc_map.emplace(
                    std::make_pair(std::string(arg3.debug_friendly_name), ArgumentDescriptor{arg3, index}));
            }
        } else {
            if (isStateOutputName(arg3.name) || isShapeTensorName(arg3.name)) {
                _outputs_desc_map.emplace(std::make_pair(std::string(arg3.name), ArgumentDescriptor{arg3, index}));

            } else {
                _outputs_desc_map.emplace(
                    std::make_pair(std::string(arg3.debug_friendly_name), ArgumentDescriptor{arg3, index}));
            }
        }
    }

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "appendGraphInitialize");
    _logger.trace("ZeroExecutor::ZeroExecutor - performing appendGraphInitialize");
    graph_command_list.appendGraphInitialize(_graph);
    _logger.trace("ZeroExecutor::ZeroExecutor - closing graph command list");
    graph_command_list.close();

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "queue_execute");
    _logger.trace("ZeroExecutor::ZeroExecutor - performing executeCommandList");
    graph_command_queue.executeCommandList(graph_command_list, fence);
    _logger.trace("ZeroExecutor::ZeroExecutor - performing hostSynchronize");
    fence.hostSynchronize();
    _logger.trace("ZeroExecutor::ZeroExecutor - hostSynchronize completed");

    if (config.has<WORKLOAD_TYPE>()) {
        setWorkloadType(config.get<WORKLOAD_TYPE>());
    }
}

void ZeroExecutor::setWorkloadType(const ov::WorkloadType workloadType) const {
    ze_command_queue_workload_type_t zeWorkloadType;
    switch (workloadType) {
    case ov::WorkloadType::DEFAULT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_DEFAULT;
        break;
    case ov::WorkloadType::EFFICIENT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_BACKGROUND;
        break;
    default:
        OPENVINO_THROW("Unknown value for WorkloadType!");
    }

    for (auto& queue : _command_queues) {
        queue->setWorkloadType(zeWorkloadType);
    }
}

void ZeroExecutor::setArgumentValue(uint32_t argi_, const void* argv_) const {
    zeroUtils::throwOnFail("zeGraphSetArgumentValue", _graph_ddi_table_ext->pfnSetArgumentValue(_graph, argi_, argv_));
}

ZeroExecutor::~ZeroExecutor() {
    auto result = _graph_ddi_table_ext->pfnDestroy(_graph);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("_graph_ddi_table_ext->pfnDestroy failed %#X", uint64_t(result));
    }
}
