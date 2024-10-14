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

ZeroExecutor::ZeroExecutor(ze_graph_handle_t graphHandle,
                           ze_device_handle_t deviceHandle,
                           ze_context_handle_t contextHandle,
                           ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                           ze_command_queue_npu_dditable_ext_curr_t& commandQueueDdiTable,
                           const Config& config,
                           uint32_t groupOrdinal)
    : _config(config),
      _logger("Graph", _config.get<LOG_LEVEL>()),
      _command_queue{std::make_shared<CommandQueue>(deviceHandle,
                                                    contextHandle,
                                                    zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                    commandQueueDdiTable,
                                                    _config,
                                                    groupOrdinal)} {
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
