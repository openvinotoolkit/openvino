// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_mem_pool.hpp"

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

using namespace intel_npu;

ZeroMemPool::ZeroMemPool() {}

ZeroMemPool& ZeroMemPool::get_instance() {
    static ZeroMemPool instance;
    return instance;
}

std::shared_ptr<ZeroMem> ZeroMemPool::allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                           const size_t bytes,
                                                           const size_t alignment,
                                                           const bool is_input) {
    _init_structs = init_structs;
    auto zero_memory = std::shared_ptr<ZeroMem>(new ZeroMem(_init_structs, bytes, alignment, is_input),
                                                [this, zero_context = _init_structs->getContext()](ZeroMem* ptr) {
                                                    delete_pool_entry(zero_context, ptr);
                                                });

    update_pool(_init_structs->getContext(), zero_memory);

    return zero_memory;
}

std::shared_ptr<ZeroMem> ZeroMemPool::import_fd_win32_zero_memory(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const size_t bytes,
    const size_t alignment,
    const void* data) {
    _init_structs = init_structs;
    auto zero_memory = std::shared_ptr<ZeroMem>(new ZeroMem(_init_structs, bytes, alignment, data),
                                                [this, zero_context = _init_structs->getContext()](ZeroMem* ptr) {
                                                    delete_pool_entry(zero_context, ptr);
                                                });

    update_pool(_init_structs->getContext(), zero_memory);

    return zero_memory;
}

std::shared_ptr<ZeroMem> ZeroMemPool::import_standard_allocation_zero_memory(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const size_t bytes,
    const size_t alignment,
    const void* data,
    const bool is_input) {
    _init_structs = init_structs;
    auto zero_memory = std::shared_ptr<ZeroMem>(new ZeroMem(_init_structs, bytes, alignment, data, is_input),
                                                [this, zero_context = _init_structs->getContext()](ZeroMem* ptr) {
                                                    delete_pool_entry(zero_context, ptr);
                                                });

    update_pool(_init_structs->getContext(), zero_memory);

    return zero_memory;
}

std::shared_ptr<ZeroMem> ZeroMemPool::get_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                      const size_t bytes,
                                                      const void* data) {
    auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(init_structs->getContext(), data);
    if (memory_id == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (_pool.find(memory_id) != _pool.end()) {
        // found one weak pointer in the pool
        // is it valid?
        auto obj = _pool.at(memory_id).lock();
        if (obj) {
            auto user_addr_end = static_cast<uint8_t*>(const_cast<void*>(data)) + bytes;
            auto host_addr_end = static_cast<uint8_t*>(obj->_ptr) + obj->_size;
            if (user_addr_end > host_addr_end) {
                throw ZeroMemException("Tensor memory range is out of bounds of the allocated host memory");
            }

            return obj;
        }
    }

    throw ZeroMemException("Failed to get zero memory from pool");
}

void ZeroMemPool::update_pool(ze_context_handle_t zero_context,
                              const std::shared_ptr<intel_npu::ZeroMem>& zero_memory) {
    auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(zero_context, zero_memory->_ptr);
    OPENVINO_ASSERT(memory_id != 0, "Failed to get memory allocation id");

    auto pair = std::make_pair(memory_id, zero_memory);

    std::lock_guard<std::mutex> lock(_mutex);
    _pool.emplace(pair);
}

void ZeroMemPool::delete_pool_entry(ze_context_handle_t zero_context, ZeroMem* ptr) {
    auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(zero_context, ptr->_ptr);

    std::lock_guard<std::mutex> lock(_mutex);
    if (_pool.at(memory_id).lock()) {
        // Don't destroy the command queue in case the shared ptr is in use!
        return;
    }
    _pool.erase(memory_id);
    // Destroy Command Queue
    delete ptr;
}
