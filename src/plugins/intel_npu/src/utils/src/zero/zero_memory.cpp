// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_memory.hpp"

#include <ze_mem_import_system_memory_ext.h>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

using namespace intel_npu;

ZeroMem::ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const size_t bytes,
                 const size_t alignment,
                 const uint32_t zero_memory_flag,
                 const void* data,
                 const bool import_fd_win32)
    : _init_structs(init_structs),
      _logger("ZeHostMem", Logger::global().level()) {
    if (import_fd_win32) {
#ifdef _WIN32
        // in the case of the Windows platform memory is locked by the D3D12 memory management - using zeMemAllocDevice
        // to import memory
        ze_external_memory_import_win32_handle_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
                                                                  nullptr,
                                                                  ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                                                                  const_cast<void*>(data),
                                                                  nullptr};
        ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, 0, 0};
        THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocDevice",
                                    zeMemAllocDevice(_init_structs->getContext(),
                                                     &desc,
                                                     bytes,
                                                     utils::STANDARD_PAGE_SIZE,
                                                     _init_structs->getDevice(),
                                                     &_ptr));
#else
        // in the case of Linux platforms memory could be changed after allocation - using zeMemAllocHost for importing
        // memory
        ze_external_memory_import_fd_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                                                        nullptr,
                                                        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                                                        static_cast<int>(reinterpret_cast<intptr_t>(data))};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, 0};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, bytes, utils::STANDARD_PAGE_SIZE, &_ptr));
#endif
    } else if (data == nullptr) {
        _size = bytes + alignment - (bytes % alignment);
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, zero_memory_flag};
        THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocHost",
                                    zeMemAllocHost(_init_structs->getContext(), &desc, _size, alignment, &_ptr));
    } else {
        _size = bytes;
        _ze_external_memory_import_system_memory_t memory_import = {
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_SYSTEM_MEMORY,
            nullptr,
            const_cast<void*>(data),
            _size};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, zero_memory_flag};
        auto result = zeMemAllocHost(_init_structs->getContext(), &desc, _size, alignment, &_ptr);

        if (result != ZE_RESULT_SUCCESS) {
            _logger.info("Importing memory through zeMemAllocHost failed, result: %s, code %#X - %s",
                         ze_result_to_string(result).c_str(),
                         uint64_t(result),
                         ze_result_to_description(result).c_str());

            throw ZeroTensorException("Importing memory failed");
        }
    }
}

ZeroMem::~ZeroMem() {
    auto result = zeMemFree(_init_structs->getContext(), _ptr);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("L0 zeMemFree result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
    }
}

ZeroMemoryPool::ZeroMemoryPool() {}

ZeroMemoryPool& ZeroMemoryPool::get_instance() {
    static ZeroMemoryPool instance;
    return instance;
}

std::shared_ptr<ZeroMem> ZeroMemoryPool::allocate_and_get_zero_memory(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const size_t bytes,
    const size_t alignment,
    const uint32_t zero_memory_flag,
    const void* data,
    const bool import_fd_win32) {
    auto zero_memory = std::shared_ptr<ZeroMem>(
        new ZeroMem(init_structs, bytes, alignment, zero_memory_flag, data),
        [this, zero_context = init_structs->getContext()](ZeroMem* ptr) {
            auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(zero_context, ptr->_ptr);

            std::lock_guard<std::mutex> lock(_mutex);
            if (_pool.at(memory_id).lock()) {
                // Don't destroy the command queue in case the shared ptr is in use!
                return;
            }
            _pool.erase(memory_id);
            // Destroy Command Queue
            delete ptr;
        });

    auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(init_structs->getContext(), zero_memory->_ptr);
    OPENVINO_ASSERT(memory_id != 0, "Failed to get memory allocation id");

    auto pair = std::make_pair(memory_id, zero_memory);

    std::lock_guard<std::mutex> lock(_mutex);
    _pool.emplace(pair);

    return zero_memory;
}

std::shared_ptr<ZeroMem> ZeroMemoryPool::get_zero_memory(const uint64_t id) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_pool.find(id) != _pool.end()) {
        // found one weak pointer in the pool
        // is it valid?
        auto obj = _pool.at(id).lock();
        if (obj) {
            return obj;
        }
    }

    return nullptr;
}
