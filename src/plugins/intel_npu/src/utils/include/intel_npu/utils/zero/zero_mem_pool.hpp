// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace intel_npu {

class ZeroInitStructsHolder;
class ZeroMem;

/**
 * @brief Per-context registry of tracked Level Zero memory objects.
 * @details Stores weak references to imported and allocated ZeroMem instances together with the synchronization
 * primitives used to serialize pool updates and coordinate pool entry removal with re-import attempts.
 */
struct ZeroMemPool {
    // Tracks Level Zero allocations and imported memory by allocation ID without extending their lifetime.
    std::unordered_map<uint64_t, std::weak_ptr<ZeroMem>> mem_pool;
    // Signals completion of asynchronous pool entry removal before a standard allocation is imported again.
    std::unordered_map<uint64_t, std::promise<void>> notify_mem_pool;
    // Serializes pool lookups and mutations.
    std::mutex mem_pool_mutex;
    // Coordinates custom deleters with standard-allocation import paths.
    std::mutex mem_pool_deleter_mutex;
};

/**
 * @details Provides helpers to either allocate or import memory in the level zero context. Such method will also add an
 * entry into the pool for the tracking ID that corresponds to that allocation. Provides a helper to check if a given
 * memory range was previously allocated or imported.
 */
namespace zero_mem {

/**
 * @brief Returns a new memory region allocated in the level zero context and adds it to the pool.
 * @param init_structs Holder for the level zero structures.
 * @param bytes Size in bytes of the memory that must be allocated.
 * @param alignment Alignment needed for the memory; it must be a multiple of the standard page size.
 * @param is_input Memory is used only as input or not.
 */
std::shared_ptr<ZeroMem> allocate_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
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
std::shared_ptr<ZeroMem> import_standard_allocation_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                           const void* data,
                                                           const size_t bytes,
                                                           const bool is_input = false);

}  // namespace zero_mem
}  // namespace intel_npu
