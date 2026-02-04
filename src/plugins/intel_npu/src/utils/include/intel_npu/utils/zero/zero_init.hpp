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
        return _driver_handle;
    }
    inline ze_device_handle_t getDevice() const {
        return _device_handle;
    }
    inline ze_context_handle_t getContext() const {
        return _context;
    }
    inline ze_graph_dditable_ext_curr_t& getGraphDdiTable() const {
        return *_graph_dditable_ext_decorator;
    }
    inline ze_command_queue_npu_dditable_ext_curr_t& getCommandQueueDdiTable() const {
        return *_command_queue_npu_dditable_ext_decorator;
    }
    inline ze_graph_profiling_dditable_ext_curr_t& getProfilingDdiTable() const {
        return *_graph_profiling_npu_dditable_ext_decorator;
    }
    inline uint32_t getDriverVersion() const {
        return _driver_properties.driverVersion;
    }
    inline uint32_t getMutableCommandListExtVersion() const {
        return _mutable_command_list_ext_version;
    }
    inline ze_api_version_t getZeDrvApiVersion() const {
        return _ze_drv_api_version;
    }
    // Helper function to check if extension with <ext_name> exists and its newer than <version>
    inline bool isExtensionSupported(std::string ext_name, uint32_t version) const {
        auto iter = _driver_extension_properties.find(ext_name);
        if (iter == _driver_extension_properties.end()) {
            return false;
        } else if (iter->second >= version) {
            return true;
        }
        return false;
    }
    inline bool isExternalMemoryStandardAllocationSupported() const {
        return _external_memory_standard_allocation_supported;
    }
    inline bool isExternalMemoryFdWin32Supported() const {
        return _external_memory_fd_win32_supported;
    }

    void setContextOptions(const uint32_t options);
    void clearContextOptions(const uint32_t options);

    static const std::shared_ptr<ZeroInitStructsHolder> getInstance();

    ze_device_graph_properties_t getCompilerProperties();

    uint32_t getCompilerVersion();

private:
    void initNpuDriver();
    void getExtensionFunctionAddress(const std::string& name, const uint32_t version, void** function_address);
    void setContextProperties();

    // keep zero_api alive until context is destroyed
    std::shared_ptr<ZeroApi> _zero_api;

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
