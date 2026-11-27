// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_intel_npu_uuid.h>

#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/common/idriver.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_capabilities.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

class ZeroDevice;

/**
 * @brief Versions of the level zero API and driver extensions that the plugin targets.
 * @details The driver negotiates each extension with the level zero driver against these versions. The defaults are
 * what the plugin uses in production; tests can lower individual versions to exercise the negotiation with older
 * drivers.
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
 * @brief The level zero driver: it finds the NPU driver, negotiates the extensions with it and holds the result.
 * @details What it holds is the driver handle, what the driver supports and the extension tables. It also creates
 * the device that lives on it. The extension tables are shared with whoever needs them, so a device that keeps them
 * does not depend on the context that was created on the driver. The capabilities still carry a few device facts
 * (queue ordinal, external-memory flags) until those move to the device.
 */
class ZeroDriver final : public IDriver, public std::enable_shared_from_this<ZeroDriver> {
public:
    explicit ZeroDriver(const ZeroInitTargets& targets = {});

    ZeroDriver(const ZeroDriver&) = delete;
    ZeroDriver& operator=(const ZeroDriver&) = delete;

    /**
     * @brief The devices on this driver. They are made on first use, the driver has to be held by a shared pointer.
     */
    const std::vector<std::shared_ptr<ZeroDevice>>& getDevices();
    /**
     * @brief The device that is used when none is asked for: the first one.
     */
    const std::shared_ptr<ZeroDevice>& getDefaultDevice();

    // IDriver
    void* getDriverHandle() const override {
        return _driver_handle;
    }
    uint32_t getDriverVersion() const override {
        return _capabilities.driverVersion;
    }
    uint32_t getApiVersion() const override {
        return static_cast<uint32_t>(_capabilities.zeApiVersion);
    }
    uint32_t getGraphExtVersion() const override {
        return _graph_dditable_ext_decorator->version();
    }
    uint32_t getCommandQueueExtVersion() const override {
        return _command_queue_npu_dditable_ext_decorator->version();
    }
    bool isLUIDExtSupported() const override {
        return _capabilities.luidExtSupported;
    }
    bool isContextExtSupported() const override {
        return _context_npu_dditable_ext_decorator->version() != 0;
    }

    // Typed access, for the level zero code
    ze_driver_handle_t getZeDriver() const {
        return _driver_handle;
    }
    const ZeroCapabilities& getCapabilities() const {
        return _capabilities;
    }
    ze_graph_dditable_ext_curr_t& getGraphDdiTable() const {
        return *_graph_dditable_ext_decorator;
    }
    ze_command_queue_npu_dditable_ext_curr_t& getCommandQueueDdiTable() const {
        return *_command_queue_npu_dditable_ext_decorator;
    }
    ze_graph_profiling_dditable_ext_curr_t& getProfilingDdiTable() const {
        return *_graph_profiling_npu_dditable_ext_decorator;
    }
    ze_context_npu_dditable_ext_curr_t& getContextDdiTable() const {
        return *_context_npu_dditable_ext_decorator;
    }

private:
    void initNpuDriver(const ze_api_version_t zeApiVersion);
    void getExtensionFunctionAddress(const std::string& name, const uint32_t version, void** function_address);

    // keep zero_api alive for as long as the driver is used
    std::shared_ptr<ZeroApi> _zero_api;

    Logger _log;

    ze_driver_handle_t _driver_handle = nullptr;
    ze_driver_properties_t _driver_properties = {};

    std::shared_ptr<ze_graph_dditable_ext_decorator> _graph_dditable_ext_decorator;
    std::shared_ptr<ze_command_queue_npu_dditable_ext_decorator> _command_queue_npu_dditable_ext_decorator;
    std::shared_ptr<ze_graph_profiling_dditable_ext_decorator> _graph_profiling_npu_dditable_ext_decorator;
    std::unique_ptr<ze_driver_npu_dditable_ext_decorator> _driver_npu_dditable_ext_decorator;
    std::shared_ptr<ze_context_npu_dditable_ext_decorator> _context_npu_dditable_ext_decorator;

    ZeroCapabilities _capabilities;

    // The devices, made on first use. The handles are only used to make them, nobody else gets them from here
    std::vector<ze_device_handle_t> _device_handles;
    std::once_flag _devices_flag;
    std::vector<std::shared_ptr<ZeroDevice>> _devices;
};

}  // namespace intel_npu
