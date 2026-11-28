// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include "intel_npu/common/icontext.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_device.hpp"
#include "intel_npu/utils/zero/zero_driver.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

namespace zero_mem {
class ZeroMemPoolManager;
}  // namespace zero_mem

/**
 * @brief A level zero context created on a driver, and the memory registry that goes with it.
 * @details The context holds the driver it was created on, so the driver outlives it. Finding the driver, negotiating
 * the extensions and the devices are the work of ZeroDriver. Until the components that use this class take narrower
 * inputs, it also hands out the driver.
 */
class ZeroContext final : public IContext {
public:
    /**
     * @brief Finds the NPU driver and creates a context on it.
     */
    explicit ZeroContext(const ZeroInitTargets& targets = {});
    /**
     * @brief Creates a context on a driver that was already found.
     */
    explicit ZeroContext(const std::shared_ptr<ZeroDriver>& driver);

    ZeroContext(const ZeroContext&) = delete;
    ZeroContext& operator=(const ZeroContext&) = delete;

    ~ZeroContext() override;

    // IContext: opaque handle, for the code that does not know the driver
    void* getContextHandle() const override {
        return _context.load();
    }

    // IContext: memory of this context, from the graph extension
    uint64_t getAllocMemSize() const override;
    uint64_t getTotalMemSize() const override;

    /**
     * @brief The driver the context was created on, and what it supports.
     */
    inline const std::shared_ptr<ZeroDriver>& getZeroDriver() const {
        return _driver;
    }

    // Typed handle, for the level zero code
    inline ze_context_handle_t getContext() const {
        return _context.load();
    }

    void setContextOptions(const uint32_t options);
    void clearContextOptions(const uint32_t options);

    static const std::shared_ptr<ZeroContext> getInstance();

    inline ZeroMemPool& getZeroMemPool() {
        return _zero_mem_pool;
    }

    static void destroyContextForInstance(std::shared_ptr<ZeroContext>& instance);

private:
    friend class zero_mem::ZeroMemPoolManager;

    void setContextProperties();
    void destroyContextLocked();

    // Declared first, so that the driver is released last
    std::shared_ptr<ZeroDriver> _driver;

    Logger _log;

    // The context handle, reset when the context is destroyed
    std::atomic<ze_context_handle_t> _context{nullptr};

    uint32_t _context_options = 0;

    // Per-context global registry for ZeroMem tracking.
    // The pool object lives in the context, while all create/import/look-up operations
    // are performed through zero_mem helper APIs (and their manager), which also synchronize
    // updates and cleanup.
    ZeroMemPool _zero_mem_pool;

    std::mutex _mutex;
};

/**
 * @brief Gets the level zero context out of the generic one that the backend hands out.
 * @details A backend other than level zero, or none, gives nullptr. The type information is shared because the backend,
 * the compiler adapters and the plugin are static libraries of the same binary.
 * @return nullptr when the given context is empty or not a level zero one.
 */
inline std::shared_ptr<ZeroContext> asZeroContext(const std::shared_ptr<IContext>& context) {
    return std::dynamic_pointer_cast<ZeroContext>(context);
}

}  // namespace intel_npu
