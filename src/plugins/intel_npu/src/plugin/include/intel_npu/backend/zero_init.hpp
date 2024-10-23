// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <memory>

#include "intel_npu/utils/logger/logger.hpp"
#include "ze_command_queue_npu_ext.h"
#include "ze_intel_npu_uuid.h"
#include "zero_types.hpp"

namespace intel_npu {
/**
 * Holder for the level zero structures which must be initialized via call to the driver once zero backend is loaded,
 * and de-initialized after their last use is over.
 */
class ZeroInitStructsHolder final {
public:
    ZeroInitStructsHolder();

    ZeroInitStructsHolder(const ZeroInitStructsHolder&) = delete;
    ZeroInitStructsHolder& operator=(const ZeroInitStructsHolder&) = delete;

    ~ZeroInitStructsHolder();

    inline ze_driver_handle_t getDriver() const {
        return driver_handle;
    }
    inline ze_device_handle_t getDevice() const {
        return device_handle;
    }
    inline ze_context_handle_t getContext() const {
        return context;
    }
    inline ze_graph_dditable_ext_curr_t& getGraphDdiTable() const {
        return *graph_dditable_ext_decorator;
    }
    inline ze_command_queue_npu_dditable_ext_curr_t& getCommandQueueDdiTable() const {
        return *command_queue_npu_dditable_ext_decorator;
    }
    inline ze_graph_profiling_dditable_ext_curr_t& getProfilingDdiTable() const {
        return *graph_profiling_npu_dditable_ext_decorator;
    }
    inline uint32_t getDriverVersion() const {
        return driver_properties.driverVersion;
    }
    inline uint32_t getMutableCommandListVersion() const {
        return mutable_command_list_version;
    }

private:
    static const ze_driver_uuid_t uuid;
    Logger log;

    ze_context_handle_t context = nullptr;
    ze_driver_handle_t driver_handle = nullptr;
    ze_device_handle_t device_handle = nullptr;

    std::unique_ptr<ze_graph_dditable_ext_decorator> graph_dditable_ext_decorator;
    std::unique_ptr<ze_command_queue_npu_dditable_ext_decorator> command_queue_npu_dditable_ext_decorator;
    std::unique_ptr<ze_graph_profiling_ddi_table_ext_decorator> graph_profiling_npu_dditable_ext_decorator;

    ze_driver_properties_t driver_properties = {};
    uint32_t mutable_command_list_version = 0;
};

}  // namespace intel_npu
