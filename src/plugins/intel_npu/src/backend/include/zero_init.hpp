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
    inline ze_graph_dditable_ext_curr_t* getGraphDdiTable() const {
        return graph_dditable_ext_decorator.get();
    }
    inline ze_command_queue_npu_dditable_ext_curr_t* getCommandQueueDdiTable() const {
        return _command_queue_npu_dditable_ext;
    }
    inline ze_graph_profiling_dditable_ext_t* getProfilingDdiTable() const {
        return _graph_profiling_ddi_table_ext;
    }
    inline uint32_t getDriverVersion() const {
        return driver_properties.driverVersion;
    }
    inline uint32_t getDriverExtVersion() const {
        return driver_ext_version;
    }
    inline uint32_t getMutableCommandListVersion() const {
        return mutable_command_list_version;
    }


    static ze_context_handle_t& getContext() {
        return context;
    }

    static void setContext(ze_context_handle_t &newContext)  {
        context=newContext;
    }

    // // -------
    // static ze_driver_handle_t& getDriverHandle() {
    //     return shared_driver_handle;
    // }

    // static void setDriverHandle(ze_driver_handle_t &newDriverHandle)  {
    //     shared_driver_handle=newDriverHandle;
    // }

    // static ze_device_handle_t& getDeviceHandle() {
    //     return shared_device_handle;
    // }

    // static void setDeviceHandle(ze_device_handle_t &newDeviceHandle)  {
    //     shared_device_handle=newDeviceHandle;
    // }
    // //------------

private:
    static const ze_driver_uuid_t uuid;
    Logger log;

    ze_driver_handle_t driver_handle = nullptr;
    ze_device_handle_t device_handle = nullptr;

    static ze_context_handle_t context;
    // static ze_driver_handle_t shared_driver_handle;
    // static ze_device_handle_t shared_device_handle;

    std::unique_ptr<ze_graph_dditable_ext_decorator> graph_dditable_ext_decorator;
    ze_command_queue_npu_dditable_ext_curr_t* _command_queue_npu_dditable_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    ze_driver_properties_t driver_properties = {};
    uint32_t driver_ext_version = 0;
    uint32_t mutable_command_list_version = 0;
};

}  // namespace intel_npu
