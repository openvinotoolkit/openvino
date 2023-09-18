// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/remote_allocators.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include <memory>

namespace ov {
namespace intel_gpu {

void* USMHostAllocator::allocate(const size_t bytes, const size_t /* alignment */) noexcept {
    try {
        ov::AnyMap params = { ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER) };
        _usm_host_tensor = _context->create_tensor(ov::element::u8, {bytes}, params);
        if (auto casted = std::dynamic_pointer_cast<RemoteTensorImpl>(_usm_host_tensor._ptr)) {
            return casted->get_original_memory()->get_internal_params().mem;
        }
        return nullptr;
    } catch (std::exception&) {
        return nullptr;
    }
}

bool USMHostAllocator::deallocate(void* /* handle */, const size_t /* bytes */, size_t /* alignment */) noexcept {
    try {
        _usm_host_tensor = {nullptr, nullptr};
    } catch (std::exception&) { }
    return true;
}

bool USMHostAllocator::is_equal(const USMHostAllocator& other) const {
    return other._usm_host_tensor != nullptr && _usm_host_tensor != nullptr && other._usm_host_tensor._ptr == _usm_host_tensor._ptr;
}
}  // namespace intel_gpu
}  // namespace ov
