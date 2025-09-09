// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_memory.hpp"

#include <ze_mem_import_system_memory_ext.h>

#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {
namespace zeroMemory {

void* HostMemAllocator::allocate(const size_t bytes, const size_t /*alignment*/) noexcept {
    size_t size = bytes + _alignment - (bytes % _alignment);

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, _flag};
    void* data = nullptr;
    auto result = zeMemAllocHost(_initStructs->getContext(), &desc, size, _alignment, &data);

    if (result == ZE_RESULT_SUCCESS) {
        return data;
    } else {
        _logger.error("L0 zeMemAllocHost result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
        return nullptr;
    }
}
bool HostMemAllocator::deallocate(void* handle, const size_t /* bytes */, size_t /* alignment */) noexcept {
    auto result = zeMemFree(_initStructs->getContext(), handle);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("L0 zeMemFree result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
        return false;
    }

    return true;
}

void* HostMemSharedAllocator::allocate(const size_t /*bytes*/, const size_t /*alignment*/) noexcept {
    _ze_external_memory_import_system_memory_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_SYSTEM_MEMORY,
                                                                nullptr,
                                                                _tensor->data(),
                                                                _tensor->get_byte_size()};

    void* data = nullptr;

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, _flag};
    auto result = zeMemAllocHost(_initStructs->getContext(), &desc, _tensor->get_byte_size(), _alignment, &data);

    if (result == ZE_RESULT_SUCCESS) {
        return data;
    } else {
        _logger.debug("Got an error when importing a CPUVA: %s, code %#X - %s\nFallback on allocating a L0 memory",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());

        return HostMemAllocator::allocate(_tensor->get_byte_size());
    }
}

}  // namespace zeroMemory
}  // namespace intel_npu
