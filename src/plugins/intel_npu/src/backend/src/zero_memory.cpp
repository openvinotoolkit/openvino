// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_memory.hpp"

#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {
namespace zeroMemory {

void* HostMemAllocator::allocate(const size_t bytes, const size_t /*alignment*/) noexcept {
    size_t size = bytes + _alignment - (bytes % _alignment);

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                     nullptr,
                                     static_cast<ze_host_mem_alloc_flags_t>(_flag)};
    void* data = nullptr;
    ze_result_t result = zeMemAllocHost(_initStructs->getContext(), &desc, size, _alignment, &data);

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
        _logger.error("L0 zeMemAllocHost result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
        return false;
    }

    return true;
}
bool HostMemAllocator::is_equal(const HostMemAllocator& other) const {
    return (_initStructs == other._initStructs) && (_flag == other._flag);
}

}  // namespace zeroMemory
}  // namespace intel_npu
