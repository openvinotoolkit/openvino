// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_mem.hpp"

#include <ze_mem_import_system_memory_ext.h>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

ZeroMem::ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const size_t bytes,
                 const size_t alignment,
                 const bool is_input)
    : _init_structs(init_structs),
      _logger("ZeHostMem", Logger::global().level()),
      _size((bytes + alignment - 1) & ~(alignment - 1)) {
    uint32_t zero_memory_flag = 0;
    if (is_input) {
        zero_memory_flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
    }
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, zero_memory_flag};
    THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocHost",
                                zeMemAllocHost(_init_structs->getContext(), &desc, _size, alignment, &_ptr));

    _id = zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(), _ptr);
    OPENVINO_ASSERT(_id != 0, "Failed to get memory allocation id of the allocated memory");
}

ZeroMem::ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const void* data,
                 const size_t bytes,
                 const bool is_input,
                 const bool standard_allocation)
    : _init_structs(init_structs),
      _logger("ZeHostMem", Logger::global().level()),
      _size(bytes) {
    if (standard_allocation) {
        if (!_init_structs->isExternalMemoryStandardAllocationSupported()) {
            throw ZeroMemException("Importing standard allocation is not supported with this driver version");
        }

        if (!utils::memory_and_size_aligned_to_standard_page_size(data, _size)) {
            throw ZeroMemException(
                "Importing standard allocation is not supported if memory is not aligned to standard page size");
        }

        // We need to check if the end of the current region is part of a previous imported region
        // The other cases are handled by the driver
        if (zeroUtils::get_l0_context_memory_allocation_id(
                _init_structs->getContext(),
                static_cast<void*>(static_cast<uint8_t*>(const_cast<void*>(data)) + _size)) > 0) {
            throw ZeroMemException("Can not import a memory which is part of an existing allocation");
        }

        uint32_t zero_memory_flag = 0;
        if (is_input) {
            zero_memory_flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
        }
        _ze_external_memory_import_system_memory_t memory_import = {
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_SYSTEM_MEMORY,
            nullptr,
            const_cast<void*>(data),
            _size};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, zero_memory_flag};
        auto result = zeMemAllocHost(_init_structs->getContext(), &desc, _size, utils::STANDARD_PAGE_SIZE, &_ptr);

        if (result != ZE_RESULT_SUCCESS) {
            _logger.info("Importing memory through zeMemAllocHost failed, result: %s, code %#X - %s",
                         ze_result_to_string(result).c_str(),
                         uint64_t(result),
                         ze_result_to_description(result).c_str());

            throw ZeroMemException("Importing memory failed");
        }
    } else {
        OPENVINO_ASSERT(_init_structs->isExternalMemoryFdWin32Supported(),
                        "Remote tensor functionality is not supported with this driver version");

        OPENVINO_ASSERT(data != nullptr, "Data pointer for importing memory can't be null");
#ifdef _WIN32
        // in the case of the Windows platform memory is locked by the D3D12 memory management - using
        // zeMemAllocDevice to import memory
        ze_external_memory_import_win32_handle_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
                                                                  nullptr,
                                                                  ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                                                                  const_cast<void*>(data),
                                                                  nullptr};
        ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, 0, 0};
        THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocDevice",
                                    zeMemAllocDevice(_init_structs->getContext(),
                                                     &desc,
                                                     _size,
                                                     utils::STANDARD_PAGE_SIZE,
                                                     _init_structs->getDevice(),
                                                     &_ptr));
#else
        // in the case of Linux platforms memory could be changed after allocation - using zeMemAllocHost for
        // importing memory
        ze_external_memory_import_fd_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                                                        nullptr,
                                                        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                                                        static_cast<int>(reinterpret_cast<intptr_t>(data))};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, 0};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, _size, utils::STANDARD_PAGE_SIZE, &_ptr));
#endif
    }

    _id = zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(), _ptr);
    OPENVINO_ASSERT(_id != 0, "Failed to get memory allocation id of the imported memory");
}

void* ZeroMem::data() {
    return _ptr;
}

size_t ZeroMem::size() {
    return _size;
}

uint64_t ZeroMem::id() {
    return _id;
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

}  // namespace intel_npu
