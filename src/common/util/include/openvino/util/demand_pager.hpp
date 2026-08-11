// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

namespace ov::util {

/**
 * @brief Populates memory regions on first touch, by delegating the faults they raise to a user callback.
 *
 * A region obtained from reserve() and passed to register_region() starts out unpopulated. The first access to any of
 * its pages suspends the accessing thread and invokes the registered callback, which is expected to fill the region
 * through populate(). The access resumes once the region is populated.
 *
 * Instances route faults only for the regions registered through them, but the underlying fault interception may be
 * process-wide, so instances are not fully isolated from each other.
 */
class DemandPager {
public:
    using pointer_type = void*;
    using size_type = std::size_t;

    /**
     * @brief Callback invoked when a registered region is first touched.
     *
     * Called with the user data and the bounds given to register_region(), never with the faulting address itself.
     * The thread it runs on is platform dependent and unspecified, and it may run concurrently for the same region,
     * so it has to be reentrant. It must not access the faulting region, as that would deadlock the fault handling.
     * It is responsible for reporting its own failures, because a region left unpopulated makes the pending access
     * either block forever or fail with a memory access error.
     */
    using callback_type = void (*)(void* user_data, pointer_type addr, size_type size) noexcept;

    DemandPager();
    ~DemandPager();

    DemandPager(const DemandPager&) = delete;
    DemandPager& operator=(const DemandPager&) = delete;
    DemandPager(DemandPager&&) = delete;
    DemandPager& operator=(DemandPager&&) = delete;

    /**
     * @brief Checks whether fault delegation is supported by the platform and the running kernel.
     * @return true if register_region() can succeed, false if every region has to be populated up front.
     */
    bool is_available() const noexcept;

    /**
     * @brief Reserves a memory region which can be registered for fault delegation.
     *
     * When is_available() is false the region is returned fully populated and reads as zeros.
     *
     * @param size Size of the region in bytes. Should be a multiple of the system page size.
     * @return Address of the region, or nullptr on failure.
     */
    pointer_type reserve(size_type size) noexcept;

    /**
     * @brief Releases a region obtained from reserve().
     * @param addr Address returned by reserve().
     * @param size Size passed to reserve().
     */
    void release(pointer_type addr, size_type size) noexcept;

    /**
     * @brief Registers a region so that faults on its pages are delegated to @p user_callback.
     *
     * @param user_callback Callback invoked when the region is touched, see callback_type for its contract.
     * @param user_data     Opaque pointer passed back to @p user_callback.
     * @param addr          Address returned by reserve().
     * @param size          Size passed to reserve().
     * @return true if the region is registered, false if delegation is unavailable or the range was refused.
     */
    bool register_region(callback_type user_callback, void* user_data, pointer_type addr, size_type size) noexcept;

    /**
     * @brief Stops delegating faults on a registered region. Does nothing if @p addr is not registered.
     * @param addr Address passed to register_region().
     */
    void unregister_region(pointer_type addr) noexcept;

    /**
     * @brief Rebinds the user data of a registered region, e.g. after its owner has been moved.
     * @param addr      Address passed to register_region().
     * @param user_data New opaque pointer to pass to the callback.
     */
    void update_user_data(pointer_type addr, void* user_data) noexcept;

    /**
     * @brief Populates a registered region with @p src, resuming every access pending on any of its pages.
     *
     * The whole region is populated at once, so @p src must be @p size bytes long. May be called from the callback.
     *
     * @param addr Address passed to register_region().
     * @param size Size passed to register_region().
     * @param src  Content to install, must not overlap the region.
     * @return true if the region is fully populated.
     */
    bool populate(pointer_type addr, size_type size, const void* src) noexcept;

    /**
     * @brief Drops the content of a registered region so that the next access faults again.
     * @param addr Address passed to register_region().
     * @param size Size passed to register_region().
     */
    void evict(pointer_type addr, size_type size) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace ov::util
