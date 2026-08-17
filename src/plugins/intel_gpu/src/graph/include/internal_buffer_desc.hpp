// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/except.hpp"

namespace cldnn {

enum class internal_buffer_size_policy {
    static_size,
    runtime_resolved,
};

enum class internal_buffer_resize_policy {
    grow_only,
    exact,
};

enum class internal_buffer_lifetime {
    primitive_instance,
};

enum class internal_buffer_access {
    read,
    write,
    read_write,
};

/// Operation-neutral internal GPU buffer requirement.
///
/// A runtime-resolved descriptor is recomputed from kernel_impl_params during a
/// dynamic-shape update. Its returned layout must be allocatable at that point.
/// The existing primitive memory pool owns the allocation for the primitive
/// instance lifetime and applies the selected resize/reuse policy.
struct BufferDescriptor {
    explicit BufferDescriptor(const layout& buffer_layout,
                              bool lockable = false,
                              bool shareable = true,
                              size_t alignment = 1,
                              internal_buffer_size_policy size_policy = internal_buffer_size_policy::static_size,
                              internal_buffer_resize_policy resize_policy = internal_buffer_resize_policy::grow_only,
                              internal_buffer_access access = internal_buffer_access::read_write)
        : m_lockable(lockable),
          m_shareable(shareable),
          m_layout(buffer_layout),
          m_alignment(alignment),
          m_size_policy(size_policy),
          m_resize_policy(resize_policy),
          m_access(access) {}

    BufferDescriptor(const ov::PartialShape& shape,
                     ov::element::Type type,
                     bool lockable = false,
                     bool shareable = true,
                     size_t alignment = 1,
                     internal_buffer_size_policy size_policy = internal_buffer_size_policy::static_size,
                     internal_buffer_resize_policy resize_policy = internal_buffer_resize_policy::grow_only,
                     internal_buffer_access access = internal_buffer_access::read_write)
        : BufferDescriptor(layout(shape, type, format::bfyx), lockable, shareable, alignment, size_policy, resize_policy, access) {}

    BufferDescriptor(size_t elements_count,
                     ov::element::Type type,
                     bool lockable = false,
                     bool shareable = true,
                     size_t alignment = 1,
                     internal_buffer_size_policy size_policy = internal_buffer_size_policy::static_size,
                     internal_buffer_resize_policy resize_policy = internal_buffer_resize_policy::grow_only,
                     internal_buffer_access access = internal_buffer_access::read_write)
        : BufferDescriptor(layout({static_cast<int64_t>(elements_count)}, type, format::bfyx),
                           lockable,
                           shareable,
                           alignment,
                           size_policy,
                           resize_policy,
                           access) {}

    void validate(size_t backend_alignment) const {
        OPENVINO_ASSERT(m_alignment != 0 && (m_alignment & (m_alignment - 1)) == 0, "[GPU] Internal buffer alignment must be a non-zero power of two");
        OPENVINO_ASSERT(m_alignment <= backend_alignment,
                        "[GPU] Internal buffer requires ",
                        m_alignment,
                        "-byte alignment, but the selected backend guarantees ",
                        backend_alignment,
                        " bytes");
        OPENVINO_ASSERT(!m_layout.is_dynamic() || m_layout.has_upper_bound(),
                        "[GPU] Runtime internal buffer layout must be resolved or bounded before allocation");
    }

    bool can_reuse_allocation(size_t allocated_bytes) const {
        const auto required_bytes = m_layout.bytes_count();
        return m_resize_policy == internal_buffer_resize_policy::grow_only ? required_bytes <= allocated_bytes : required_bytes == allocated_bytes;
    }

    bool m_lockable = false;
    bool m_shareable = true;
    layout m_layout;
    size_t m_alignment = 1;
    internal_buffer_size_policy m_size_policy = internal_buffer_size_policy::static_size;
    internal_buffer_resize_policy m_resize_policy = internal_buffer_resize_policy::grow_only;
    internal_buffer_lifetime m_lifetime = internal_buffer_lifetime::primitive_instance;
    internal_buffer_access m_access = internal_buffer_access::read_write;
};

}  // namespace cldnn
