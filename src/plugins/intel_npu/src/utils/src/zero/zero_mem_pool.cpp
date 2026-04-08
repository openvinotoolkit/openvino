// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_mem_pool.hpp"

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
#    include <cstdlib>
#endif

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {
namespace zero_mem {

class ZeroMemPoolManager final {
public:
    ZeroMemPoolManager() = default;
    ~ZeroMemPoolManager() = default;

    static std::shared_ptr<ZeroMem> allocate_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                    const size_t bytes,
                                                    const size_t alignment,
                                                    const bool is_input) {
        auto zero_memory = std::shared_ptr<ZeroMem>(new ZeroMem(init_structs, bytes, alignment, is_input),
                                                    [init_structs](ZeroMem* ptr) {
                                                        ZeroMemPoolManager::delete_pool_entry(init_structs, ptr);
                                                    });

        std::lock_guard<std::mutex> lock(init_structs->getZeroMemPool().mem_pool_mutex);
        {
            // lock the deleter_lock only for this scope
            std::lock_guard<std::mutex> deleter_lock(init_structs->getZeroMemPool().mem_pool_deleter_mutex);
            ZeroMemPoolManager::update_pool(init_structs, zero_memory);
        }

        return zero_memory;
    }

    static std::shared_ptr<ZeroMem> import_shared_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                         const void* data,
                                                         const size_t bytes) {
        auto zero_memory = std::shared_ptr<ZeroMem>(new ZeroMem(init_structs, data, bytes, false, false),
                                                    [init_structs](ZeroMem* ptr) {
                                                        ZeroMemPoolManager::delete_pool_entry(init_structs, ptr);
                                                    });

        std::lock_guard<std::mutex> lock(init_structs->getZeroMemPool().mem_pool_mutex);
        {
            // lock the deleter_lock only for this scope
            std::lock_guard<std::mutex> deleter_lock(init_structs->getZeroMemPool().mem_pool_deleter_mutex);
            ZeroMemPoolManager::update_pool(init_structs, zero_memory);
        }

        return zero_memory;
    }

    static std::shared_ptr<ZeroMem> import_standard_allocation_memory(
        const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
        const void* data,
        const size_t bytes,
        const bool is_input) {
        std::lock_guard<std::mutex> lock(init_structs->getZeroMemPool().mem_pool_mutex);
        std::unique_lock<std::mutex> deleter_lock(init_structs->getZeroMemPool().mem_pool_deleter_mutex);

        auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(init_structs->getContext(), data);
        if (memory_id == 0) {
            // try to import memory if it isn't part of the same zero context
            return import_standard_allocation(init_structs, data, bytes, is_input);
        }

        auto& zero_mem_pool = init_structs->getZeroMemPool().mem_pool;
        auto& notify_zero_mem_pool = init_structs->getZeroMemPool().notify_mem_pool;

        if (zero_mem_pool.find(memory_id) != zero_mem_pool.end()) {
            // found one weak pointer in the pool; check it if it's valid
            auto obj = zero_mem_pool.at(memory_id).lock();
            if (obj) {
                auto user_addr_end = static_cast<uint8_t*>(const_cast<void*>(data)) + bytes;
                auto host_addr_end = static_cast<uint8_t*>(obj->data()) + obj->size();
                if (user_addr_end > host_addr_end) {
                    throw ZeroMemException("Tensor memory range is out of bounds of the allocated host memory");
                }

                return obj;
            } else {
                // shared_ptr counter is 0, we can not lock memory; wait until the deleter frees the memory
                std::shared_future<void> done_future = notify_zero_mem_pool[memory_id].get_future().share();
                // allow (any) deleter to be executed. it will remove the pool entry and deallocate the level zero
                // memory
                deleter_lock.unlock();
                // waiting for the corresponding deleter to be executed
                done_future.wait();
                deleter_lock.lock();
                notify_zero_mem_pool.erase(memory_id);

                // import again memory after make sure it is destroyed
                return ZeroMemPoolManager::import_standard_allocation(init_structs, data, bytes, is_input);
            }
        }

        throw ZeroMemException("Unexpected error");
    }

private:
    static std::shared_ptr<ZeroMem> import_standard_allocation(
        const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
        const void* data,
        const size_t bytes,
        const bool is_input) {
        auto zero_memory = std::shared_ptr<ZeroMem>(
            new ZeroMem(init_structs, data, bytes, is_input, true),
            [init_structs](ZeroMem* ptr) {
                auto memory_id = ptr->id();
                auto& notify_zero_mem_pool = init_structs->getZeroMemPool().notify_mem_pool;
                ZeroMemPoolManager::delete_pool_entry(init_structs, ptr);

                std::unique_lock<std::mutex> deleter_lock(init_structs->getZeroMemPool().mem_pool_deleter_mutex);
                if (notify_zero_mem_pool.find(memory_id) != notify_zero_mem_pool.end()) {
                    notify_zero_mem_pool.at(memory_id).set_value();
                }
            });

        ZeroMemPoolManager::update_pool(init_structs, zero_memory);
        return zero_memory;
    }

    static void update_pool(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                            const std::shared_ptr<ZeroMem>& zero_memory) {
        auto& zero_mem_pool = init_structs->getZeroMemPool().mem_pool;
        auto pair = std::make_pair(zero_memory->id(), zero_memory);

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
        if (zero_mem_pool.find(zero_memory->id()) != zero_mem_pool.end()) {
            if (zero_mem_pool.at(zero_memory->id()).lock()) {
                // Abort instead of throw: this is a programming-error invariant violation.
                // Throwing here could cause a deadlock if the caller holds
                // _zero_mem_pool_deleter_mutex, because stack unwinding would destroy
                // zero_memory whose custom deleter re-acquires the same mutex.
                Logger log("ZeroMemPool", Logger::global().level());
                log.error("Memory exists, at this point id shall not be used!");
                std::abort();
            }
        }
#endif

        zero_mem_pool.emplace(pair);
    }

    static void delete_pool_entry(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, ZeroMem* zero_memory) {
        auto& zero_mem_pool = init_structs->getZeroMemPool().mem_pool;
        std::unique_lock<std::mutex> deleter_lock(init_structs->getZeroMemPool().mem_pool_deleter_mutex);
        zero_mem_pool.erase(zero_memory->id());
        delete zero_memory;
    }
};

std::shared_ptr<ZeroMem> allocate_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                         const size_t bytes,
                                         const size_t alignment,
                                         const bool is_input) {
    return ZeroMemPoolManager::allocate_memory(init_structs, bytes, alignment, is_input);
}

std::shared_ptr<ZeroMem> import_shared_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                              const void* data,
                                              const size_t bytes) {
    return ZeroMemPoolManager::import_shared_memory(init_structs, data, bytes);
}

std::shared_ptr<ZeroMem> import_standard_allocation_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                           const void* data,
                                                           const size_t bytes,
                                                           const bool is_input) {
    return ZeroMemPoolManager::import_standard_allocation_memory(init_structs, data, bytes, is_input);
}

}  // namespace zero_mem
}  // namespace intel_npu
