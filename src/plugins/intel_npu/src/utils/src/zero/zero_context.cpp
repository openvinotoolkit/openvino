// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_context.hpp"

#include <ze_context_npu_ext.h>
#include <ze_graph_ext.h>

#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

ZeroContext::ZeroContext(const ZeroInitTargets& targets) : ZeroContext(std::make_shared<ZeroDriver>(targets)) {}

ZeroContext::ZeroContext(const std::shared_ptr<ZeroDriver>& driver)
    : _driver(driver),
      _log("NPUZeroContext", Logger::global().level()) {
    // Create context - share between the compiler and the backend
    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0};
    ze_context_handle_t ctx = nullptr;
    THROW_ON_FAIL_FOR_LEVELZERO("zeContextCreate", zeContextCreate(_driver->getZeDriver(), &context_desc, &ctx));
    _context.store(ctx);
    _log.debug("ZeroContext initialize complete");
}

const std::shared_ptr<ZeroContext> ZeroContext::getInstance() {
    static std::mutex mutex;
    static std::weak_ptr<ZeroContext> weak_instance;

    std::lock_guard<std::mutex> lock(mutex);
    auto instance = weak_instance.lock();

    if (instance) {
        // Check if device is still valid
        auto res = zeDeviceGetStatus(instance->getZeroDriver()->getDefaultDevice()->getZeDevice());
        if (res == ZE_RESULT_ERROR_DEVICE_LOST) {
            destroyContextForInstance(instance);
            weak_instance.reset();
            instance.reset();
        }
    }

    if (!instance) {
        instance = std::make_shared<ZeroContext>();
        weak_instance = instance;
    }

    return instance;
}

void ZeroContext::destroyContextForInstance(std::shared_ptr<ZeroContext>& instance) {
    if (!instance) {
        return;
    }

    std::lock_guard<std::mutex> lock(instance->_mutex);
    instance->destroyContextLocked();
}

uint64_t ZeroContext::getAllocMemSize() const {
    ze_graph_memory_query_t query{};
    ze_result_t result =
        _driver->getGraphDdiTable().pfnQueryContextMemory(getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryContextMemory", result, _driver->getGraphDdiTable());

    return query.allocated;
}

uint64_t ZeroContext::getTotalMemSize() const {
#define LEGACY_MAX_MEM_ALLOC_SIZE_BYTES (2147483648)  // 2GB in base-2

    ze_graph_memory_query_t query{};
    ze_result_t result =
        _driver->getGraphDdiTable().pfnQueryContextMemory(getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryContextMemory", result, _driver->getGraphDdiTable());

    // For drivers with graph_extension < 1.9 we report fixed 2GB max allocation size (old drivers don't support more)
    // For drivers with graph_extension > 1.9 we report the value they return
    const uint32_t graph_ext_version = _driver->getGraphDdiTable().version();
    if (graph_ext_version >= ZE_MAKE_VERSION(1, 9)) {
        // we are safe here, can return the value directly from driver
        return query.total;
    }
#if defined(_WIN32) || defined(__CYGWIN__)
    // Special case for windows drivers with graph_extension v 1.8
    if (graph_ext_version == ZE_MAKE_VERSION(1, 8)) {
        // query here returns total system memory in KB, which we need to
        // divide by 2 (OS limitation) and convert to bytes
        return (query.total << 9);
    }
#endif

    // Default for older drivers: return 2GB
    return LEGACY_MAX_MEM_ALLOC_SIZE_BYTES;
}

void ZeroContext::setContextProperties() {
    ze_context_properties_npu_ext_t context_properties_npu_ext = {ZE_STRUCTURE_TYPE_CONTEXT_PROPERTIES_NPU_EXT,
                                                                  nullptr,
                                                                  _context_options};
    THROW_ON_FAIL_FOR_LEVELZERO(
        "pfnSetProperties",
        _driver->getContextDdiTable().pfnSetProperties(_context.load(), &context_properties_npu_ext));
}

void ZeroContext::clearContextOptions(const uint32_t options) {
    std::lock_guard<std::mutex> lock(_mutex);
    _context_options &= ~options;
    setContextProperties();
}

void ZeroContext::setContextOptions(const uint32_t options) {
    std::lock_guard<std::mutex> lock(_mutex);
    _context_options |= options;
    setContextProperties();
}

void ZeroContext::destroyContextLocked() {
    _zero_mem_pool.clear();

    if (!_context.load()) {
        return;
    }

    _log.debug("ZeroContext - performing zeContextDestroy");
    auto result = zeContextDestroy(_context.load());
    if (result == ZE_RESULT_SUCCESS) {
        _context.store(nullptr);
        _log.debug("ZeroContext - zeContextDestroy succeeded");
    } else if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
        _context.store(nullptr);
        _log.warning("zeContextDestroy failed to destroy the context; Level zero context was already destroyed");
    } else {
        _log.error("zeContextDestroy failed %#X", uint64_t(result));
    }
}

ZeroContext::~ZeroContext() {
    std::lock_guard<std::mutex> lock(_mutex);
    try {
        destroyContextLocked();
    } catch (const std::exception& e) {
        _log.error("ZeroContext::~ZeroContext - exception during context destruction: %s",
                   e.what());
    } catch (...) {
        _log.error("ZeroContext::~ZeroContext - unknown exception during context destruction");
    }
}

}  // namespace intel_npu
