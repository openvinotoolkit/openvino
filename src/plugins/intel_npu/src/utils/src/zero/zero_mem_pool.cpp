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
    auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(init_structs->getContext(), data);
    if (memory_id == 0) {
        // try to import memory if it isn't part of the same zero context
        auto zero_memory =
            std::shared_ptr<ZeroMem>(new ZeroMem(init_structs, data, bytes, false, true), [this](ZeroMem* ptr) {
                delete_pool_entry(ptr);
            });

        update_pool(zero_memory);

        return zero_memory;
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
        }
    }

    throw ZeroMemException("Failed to get zero memory from pool");
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
    std::lock_guard<std::mutex> lock(_mutex);
    _pool.erase(zero_memory->id());
    delete zero_memory;
}

}  // namespace intel_npu
