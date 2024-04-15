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
                                                      _config,
                                                      group_ordinal),
                       std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                                      _initStructs->getContext(),
                                                      zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                      _config,
                                                      group_ordinal),
                       std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                                      _initStructs->getContext(),
                                                      zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                      _config,
                                                      group_ordinal)}} {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::ZeroExecutor");
    CommandList graph_command_list(_initStructs->getDevice(),
                                   _initStructs->getContext(),
                                   _initStructs->getGraphDdiTable(),
                                   _config,
                                   _group_ordinal);
    CommandQueue graph_command_queue(_initStructs->getDevice(),
                                     _initStructs->getContext(),
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                     _config,
                                     _group_ordinal);
    Fence fence(graph_command_queue, _config);
    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDeviceGetProperties(_initStructs->getDevice(), &properties));

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

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetArgumentProperties");
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index) {
        ze_graph_argument_properties_t arg;
        zeroUtils::throwOnFail("pfnGetArgumentProperties",
                               _graph_ddi_table_ext->pfnGetArgumentProperties(_graph, index, &arg));
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            _inputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        } else {
            _outputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        }
    }
    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "appendGraphInitialize");
    graph_command_list.appendGraphInitialize(_graph);
    graph_command_list.close();

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "queue_execute");
    graph_command_queue.executeCommandList(graph_command_list, fence);
    fence.hostSynchronize();
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
