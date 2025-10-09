// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_mem_pool.hpp"

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

ZeroMemPool::ZeroMemPool() {}

ZeroMemPool& ZeroMemPool::get_instance() {
    static ZeroMemPool instance;
    return instance;
}

std::shared_ptr<ZeroMem> ZeroMemPool::allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                           const size_t bytes,
                                                           const size_t alignment,
                                                           const bool is_input) {
    auto zero_memory =
        std::shared_ptr<ZeroMem>(new ZeroMem(init_structs, bytes, alignment, is_input), [this](ZeroMem* ptr) {
            delete_pool_entry(ptr);
        });

    std::lock_guard<std::mutex> lock(_mutex);
    update_pool(zero_memory);

    return zero_memory;
}

std::shared_ptr<ZeroMem> ZeroMemPool::import_shared_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                           const void* data,
                                                           const size_t bytes) {
    auto zero_memory =
        std::shared_ptr<ZeroMem>(new ZeroMem(init_structs, data, bytes, false, false), [this](ZeroMem* ptr) {
            delete_pool_entry(ptr);
        });

    std::lock_guard<std::mutex> lock(_mutex);
    update_pool(zero_memory);

    return zero_memory;
}

std::shared_ptr<ZeroMem> ZeroMemPool::import_standard_allocation_memory(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const void* data,
    const size_t bytes,
    const bool is_input) {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unique_lock<std::mutex> deleter_lock(_deleter_mutex);

    auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(init_structs->getContext(), data);
    if (memory_id == 0) {
        // try to import memory if it isn't part of the same zero context
        return import_standard_allocation(init_structs, data, bytes, is_input);
    }

    if (_pool.find(memory_id) != _pool.end()) {
        // found one weak pointer in the pool; check it if it's valid
        auto obj = _pool.at(memory_id).lock();
        if (obj) {
            auto user_addr_end = static_cast<uint8_t*>(const_cast<void*>(data)) + bytes;
            auto host_addr_end = static_cast<uint8_t*>(obj->data()) + obj->size();
            if (user_addr_end > host_addr_end) {
                throw ZeroMemException("Tensor memory range is out of bounds of the allocated host memory");
            }

            return obj;
        } else {
            // shared_ptr counter is 0, we can not lock memory; wait until the deleter frees the memory
            std::shared_future<void> done_future = _notify_pool[memory_id].get_future().share();
            // allow (any) deleter to be executed. it will remove the pool entry and deallocate the level zero memory
            deleter_lock.unlock();
            // waiting for the corresponding deleter to be executed
            done_future.wait();
            deleter_lock.lock();
            _notify_pool.erase(memory_id);

            // import again memory after make sure it is destroyed
            return import_standard_allocation(init_structs, data, bytes, is_input);
        }
    }

    throw ZeroMemException("Unexpected error");
}

std::shared_ptr<ZeroMem> ZeroMemPool::import_standard_allocation(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const void* data,
    const size_t bytes,
    const bool is_input) {
    auto zero_memory =
        std::shared_ptr<ZeroMem>(new ZeroMem(init_structs, data, bytes, is_input, true), [this](ZeroMem* ptr) {
            auto memory_id = ptr->id();
            delete_pool_entry(ptr);
            if (_notify_pool.find(memory_id) != _notify_pool.end()) {
                _notify_pool.at(memory_id).set_value();
            }
        });

    update_pool(zero_memory);

    return zero_memory;
}

void ZeroMemPool::update_pool(const std::shared_ptr<intel_npu::ZeroMem>& zero_memory) {
    auto pair = std::make_pair(zero_memory->id(), zero_memory);

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    if (_pool.find(zero_memory->id()) != _pool.end()) {
        if (_pool.at(zero_memory->id()).lock()) {
            OPENVINO_THROW("Memory exists, at this point id shall not be used!");
        }
    }
#endif

    _pool.emplace(pair);
}

void ZeroMemPool::delete_pool_entry(ZeroMem* zero_memory) {
    std::unique_lock lock(_deleter_mutex);
    _pool.erase(zero_memory->id());
    delete zero_memory;
}

}  // namespace intel_npu
