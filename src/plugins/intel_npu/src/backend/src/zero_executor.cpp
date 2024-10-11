// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_executor.hpp"

#include <ze_api.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "ze_command_queue_npu_ext.h"
#include "zero_device.hpp"

using namespace intel_npu;

ZeroExecutor::ZeroExecutor(const std::shared_ptr<const ZeroInitStructsHolder>& initStructs,
                           const std::shared_ptr<IGraph>& graph,
                           const Config& config,
                           uint32_t group_ordinal)
    : _config(config),
      _logger("Graph", _config.get<LOG_LEVEL>()),
      _initStructs(initStructs),
      _graph(graph),
      _command_queue{std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                                    _initStructs->getContext(),
                                                    zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                    _initStructs->getCommandQueueDdiTable(),
                                                    _config,
                                                    group_ordinal)} {
    _logger.debug("ZeroExecutor::ZeroExecutor - create graph");
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_GRAPH, itt::domains::LevelZeroBackend, "Executor::ZeroExecutor", "graphCreate");

    _logger.debug("reuse graph handle created from compiler");
    auto graph_handle = static_cast<ze_graph_handle_t>(_graph->get_handle());

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetProperties");
    _logger.debug("performing pfnGetProperties");
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;

    zeroUtils::throwOnFail("pfnGetProperties", _initStructs->getGraphDdiTable().pfnGetProperties(graph_handle, &props));
    if (_initStructs->getGraphDdiTable().version() <= ZE_GRAPH_EXT_VERSION_1_1) {
        OPENVINO_THROW("Incompatibility between the NPU plugin and driver! The driver version is too old, please "
                       "update the driver version");
    }

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetArgumentProperties3");
    _logger.debug("performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        zeroUtils::throwOnFail("pfnGetArgumentProperties3",
                               _initStructs->getGraphDdiTable().pfnGetArgumentProperties3(graph_handle, index, &arg3));

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _input_descriptors.push_back(ArgumentDescriptor{arg3, index});
        } else {
            _output_descriptors.push_back(ArgumentDescriptor{arg3, index});
        }
    }

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

    _command_queue->setWorkloadType(zeWorkloadType);
}

void ZeroExecutor::mutexLock() const {
    _mutex.lock();
}

void ZeroExecutor::mutexUnlock() const {
    _mutex.unlock();
}

ZeroExecutor::~ZeroExecutor() {
    _logger.debug("~ZeroExecutor()");
}
