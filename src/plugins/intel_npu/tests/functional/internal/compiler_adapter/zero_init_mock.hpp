// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_command_queue_npu_ext.h>
#include <ze_graph_ext.h>
#include <ze_intel_npu_uuid.h>

#include <memory>
#include <mutex>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

struct ZeroInitStructsMock {
    ZeroInitStructsMock(int extVersion);

    ~ZeroInitStructsMock();

    void initNpuDriver();

    std::shared_ptr<intel_npu::ZeroApi> zero_api;

    intel_npu::Logger log;

    ze_context_handle_t context = nullptr;
    ze_driver_handle_t driver_handle = nullptr;
    ze_device_handle_t device_handle = nullptr;

    std::map<std::string, uint32_t> driver_extension_properties;
    std::unique_ptr<ze_graph_dditable_ext_decorator> graph_dditable_ext_decorator;
    std::unique_ptr<ze_command_queue_npu_dditable_ext_decorator> command_queue_npu_dditable_ext_decorator;
    std::unique_ptr<ze_graph_profiling_dditable_ext_decorator> graph_profiling_npu_dditable_ext_decorator;

    ze_driver_properties_t driver_properties = {};
    uint32_t mutable_command_list_ext_version = 0;

    ze_api_version_t ze_drv_api_version = {};

    std::unique_ptr<ze_device_graph_properties_t> compiler_properties = nullptr;

    bool _external_memory_standard_allocation_supported = false;
    bool _external_memory_fd_win32_supported = false;

    std::mutex _mutex;
};

}  // namespace intel_npu
