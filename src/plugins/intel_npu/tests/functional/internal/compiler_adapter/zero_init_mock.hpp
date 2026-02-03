// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_intel_npu_uuid.h>

#include <memory>
#include <mutex>
#include <optional>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

struct ZeroInitStructsMock {
    ZeroInitStructsMock(int extVersion);

    ~ZeroInitStructsMock();

    void initNpuDriver();
    void getExtensionFunctionAddress(const std::string& name, const uint32_t version, void** function_address);

    std::shared_ptr<intel_npu::ZeroApi> _zero_api;

    Logger _log;

    ze_context_handle_t _context = nullptr;
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;

    std::map<std::string, uint32_t> _driver_extension_properties;
    std::unique_ptr<ze_graph_dditable_ext_decorator> _graph_dditable_ext_decorator;
    std::unique_ptr<ze_command_queue_npu_dditable_ext_decorator> _command_queue_npu_dditable_ext_decorator;
    std::unique_ptr<ze_graph_profiling_dditable_ext_decorator> _graph_profiling_npu_dditable_ext_decorator;
    std::unique_ptr<ze_driver_npu_dditable_ext_decorator> _driver_npu_dditable_ext_decorator;
    std::unique_ptr<ze_context_npu_dditable_ext_decorator> _context_npu_dditable_ext_decorator;

    ze_driver_properties_t _driver_properties = {};
    uint32_t _mutable_command_list_ext_version = 0;

    ze_api_version_t _ze_drv_api_version = {};

    std::optional<ze_device_graph_properties_t> _compiler_properties = std::nullopt;

    bool _external_memory_standard_allocation_supported = false;
    bool _external_memory_fd_win32_supported = false;

    uint32_t _context_options = 0;

    std::mutex _mutex;
};

}  // namespace intel_npu
