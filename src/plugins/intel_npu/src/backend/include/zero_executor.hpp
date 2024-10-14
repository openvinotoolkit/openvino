// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <mutex>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/properties.hpp"
#include "zero_init.hpp"
#include "zero_wrappers.hpp"

namespace intel_npu {

class ZeroExecutor final : public IExecutor {
public:
    ZeroExecutor(ze_graph_handle_t graphHandle,
                 ze_device_handle_t deviceHandle,
                 ze_context_handle_t contextHandle,
                 ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                 ze_command_queue_npu_dditable_ext_curr_t& commandQueueDdiTable,
                 const Config& config,
                 uint32_t groupOrdinal);

    ZeroExecutor(const ZeroExecutor&) = delete;
    ZeroExecutor& operator=(const ZeroExecutor&) = delete;

    ~ZeroExecutor() override{};

    void setWorkloadType(const ov::WorkloadType workloadType) const override;
    void mutexLock() const;
    void mutexUnlock() const;

    inline const std::shared_ptr<CommandQueue>& getCommandQueue() const {
        return _command_queue;
    }

private:
    const Config _config;
    Logger _logger;

    std::shared_ptr<CommandQueue> _command_queue;

    mutable std::mutex _mutex;
};

}  // namespace intel_npu
