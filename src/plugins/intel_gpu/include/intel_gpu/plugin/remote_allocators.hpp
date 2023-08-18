// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/so_ptr.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

class RemoteTensorImpl;
class RemoteContextImpl;

class USMHostAllocator final {
private:
    ov::SoPtr<RemoteTensorImpl> _usm_host_tensor = { nullptr, nullptr };
    std::shared_ptr<RemoteContextImpl> _context = nullptr;

public:
    using Ptr = std::shared_ptr<USMHostAllocator>;

    USMHostAllocator(std::shared_ptr<RemoteContextImpl> context) : _context(context) { }

    /**
    * @brief Allocates memory
    * @param size The size in bytes to allocate
    * @return Handle to the allocated resource
    */
    void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t)) noexcept;
    /**
    * @brief Releases handle and all associated memory resources which invalidates the handle.
    * @return false if handle cannot be released, otherwise - true.
    */
    bool deallocate(void* handle, const size_t bytes, size_t alignment = alignof(max_align_t)) noexcept;

    bool is_equal(const USMHostAllocator& other) const;
};

}  // namespace intel_gpu
}  // namespace ov
