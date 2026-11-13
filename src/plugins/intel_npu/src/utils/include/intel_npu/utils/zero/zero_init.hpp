// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_intel_npu_uuid.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_capabilities.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

namespace zero_mem {
class ZeroMemPoolManager;
}  // namespace zero_mem

/**
 * @brief Versions of the level zero API and driver extensions that the plugin targets.
 * @details The holder negotiates each extension with the driver against these versions. The defaults are what the
 * plugin uses in production; tests can lower individual versions to exercise the negotiation with older drivers.
 */
struct ZeroInitTargets {
    ze_api_version_t zeApiVersion = ZE_API_VERSION_1_18;
    uint32_t driverNpuExtVersion = ZE_DRIVER_NPU_EXT_VERSION_1_0;
    uint32_t graphNpuExtVersion = ZE_GRAPH_EXT_VERSION_1_19;
    uint32_t commandQueueNpuExtVersion = ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_1;
    uint32_t profilingNpuExtVersion = ZE_PROFILING_DATA_EXT_VERSION_1_0;
    uint32_t contextNpuExtVersion = ZE_CONTEXT_NPU_EXT_VERSION_1_0;
    uint32_t mutableCommandListExtVersion = ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1;
    uint32_t externalMemMapSysMemExtVersion = ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_1_0;
    uint32_t commandQueueSetPriorityExtVersion = ZE_COMMAND_QUEUE_SET_PRIORITY_EXT_VERSION_1_0;
    uint32_t deviceLuidExtVersion = ZE_DEVICE_LUID_EXT_VERSION_1_0;
};

/**
 * Holder for the level zero structures which must be initialized via call to the driver once zero backend is loaded,
 * and de-initialized after their last use is over.
 */
class ZeroInitStructsHolder final {
public:
    explicit ZeroInitStructsHolder(const ZeroInitTargets& targets = {});

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
        return _context.load();
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
    inline uint32_t getGraphExtVersion() const {
        return _graph_dditable_ext_decorator->version();
    }
    inline uint32_t getMutableCommandListExtVersion() const {
        return _capabilities.mutableCommandListExtVersion;
    }
    inline ze_api_version_t getZeDrvApiVersion() const {
        return _capabilities.zeApiVersion;
    }
    inline bool isCommandQueueExtSupported() const {
        return _capabilities.commandQueueExtSupported;
    }
    inline bool isLUIDExtSupported() const {
        return _capabilities.luidExtSupported;
    }
    inline bool isContextExtSupported() const {
        return _capabilities.contextExtSupported;
    }
    inline bool isExternalMemoryStandardAllocationSupported() const {
        return _capabilities.externalMemoryStandardAllocationSupported;
    }
    inline bool isExternalMemoryFdWin32Supported() const {
        return _capabilities.externalMemoryFdWin32Supported;
    }
    inline bool isCommandQueueSetPrioritySupported() const {
        return _capabilities.commandQueueSetPrioritySupported;
    }
    inline uint32_t getCommandQueueGroupOrdinal() const {
        return _capabilities.commandQueueGroupOrdinal;
    }

    /**
     * @brief What the driver supports, as a plain value. Fixed once the driver has been queried.
     */
    inline const ZeroCapabilities& getCapabilities() const {
        return _capabilities;
    }

    void setContextOptions(const uint32_t options);
    void clearContextOptions(const uint32_t options);

    static const std::shared_ptr<ZeroInitStructsHolder> getInstance();

    ze_device_graph_properties_t getCompilerProperties();

    uint32_t getCompilerVersion();

    inline ZeroMemPool& getZeroMemPool() {
        return _zero_mem_pool;
    }

    static void destroyContextForInstance(std::shared_ptr<ZeroInitStructsHolder>& instance);

private:
    friend class zero_mem::ZeroMemPoolManager;

    void initNpuDriver(const ze_api_version_t zeApiVersion);
    void initCompilerPropertiesLocked();
    void getExtensionFunctionAddress(const std::string& name, const uint32_t version, void** function_address);
    void setContextProperties();
    void destroyContextLocked();

    // keep zero_api alive until context is destroyed
    std::shared_ptr<ZeroApi> _zero_api;

    Logger _log;

    std::atomic<ze_context_handle_t> _context{nullptr};
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;

    std::unique_ptr<ze_graph_dditable_ext_decorator> _graph_dditable_ext_decorator;
    std::unique_ptr<ze_command_queue_npu_dditable_ext_decorator> _command_queue_npu_dditable_ext_decorator;
    std::unique_ptr<ze_graph_profiling_dditable_ext_decorator> _graph_profiling_npu_dditable_ext_decorator;
    std::unique_ptr<ze_driver_npu_dditable_ext_decorator> _driver_npu_dditable_ext_decorator;
    std::unique_ptr<ze_context_npu_dditable_ext_decorator> _context_npu_dditable_ext_decorator;

    ze_driver_properties_t _driver_properties = {};
    std::optional<ze_device_graph_properties_t> _compiler_properties = std::nullopt;

    uint32_t _context_options = 0;

    ZeroCapabilities _capabilities;

    // Per-context global registry for ZeroMem tracking.
    // The pool object lives in the context holder, while all create/import/look-up operations
    // are performed through zero_mem helper APIs (and their manager), which also synchronize
    // updates and cleanup.
    ZeroMemPool _zero_mem_pool;

    std::mutex _mutex;
};

}  // namespace intel_npu
