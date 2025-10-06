// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"

namespace intel_npu {

/**
 * @brief ZeroMemPool API used to keep track of all allocations and memory imports performed in the level zero context.
 * @details Provides methods to either allocate or import memory in the level zero context. Such method will also add an
 * entry into the pool for the tracking ID that corresponds to that allocation. Provides a method to check if a given
 * memory range was previously.
 */
class ZeroMemPool final {
public:
    ZeroMemPool();
    ZeroMemPool(const ZeroMemPool& other) = delete;
    ZeroMemPool(ZeroMemPool&& other) = delete;
    void operator=(const ZeroMemPool&) = delete;
    void operator=(ZeroMemPool&&) = delete;

    /**
     * @brief Get static instance.
     */
    static ZeroMemPool& get_instance();

    /**
     * @brief Returns a new memory region allocated in the level zero and adds it to the pool.
     * @param init_structs Holder for the level zero structures.
     * @param bytes Size in bytes of the memory that must be allocated.
     * @param alignment Alignment needed for the memory; it must be a multiple of the standard page size.
     * @param is_input Memory is used only as input or not.
     */
    std::shared_ptr<ZeroMem> allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                  const size_t bytes,
                                                  const size_t alignment,
                                                  const bool is_input = false);

    /**
     * @brief Returns an imported shared(CMX-DMA in case of Linux, NT handle in case of Windows) memory in the level
     * zero and adds it to the pool.
     * @param init_structs Holder for the level zero structures.
     * @param data Memory to be imported.
     * @param bytes Size in bytes of the memory that must be allocated.
     */
    std::shared_ptr<ZeroMem> import_shared_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                  const void* data,
                                                  const size_t bytes);

    /**
     * @brief Performs a look-up in the pool to check if the entire range given by [data, data + bytes] was previously
     * allocated or imported. Returns a reference to the ZeroMem object stored in the pool in case the check is
     * successful, or tries to import memory in the level zero context, adds it to the pool, and returns it.
     * @param init_structs Holder for the level zero structures.
     * @param data User memory to be checked.
     * @param bytes Size in bytes of the memory.
     * @param is_input Memory is used only as input or not.
     */
    std::shared_ptr<ZeroMem> import_standard_allocation_memory(
        const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
        const void* data,
        const size_t bytes,
        const bool is_input = false);

private:
    void update_pool(const std::shared_ptr<intel_npu::ZeroMem>& zero_memory);
    void delete_pool_entry(ZeroMem* ptr);

    std::unordered_map<uint64_t, std::weak_ptr<ZeroMem>> _pool;
    std::mutex _mutex;
};

}  // namespace intel_npu
