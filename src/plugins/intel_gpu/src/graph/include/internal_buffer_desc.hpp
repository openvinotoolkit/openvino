// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"

namespace cldnn {

/// Operation-neutral internal GPU buffer requirement.
struct BufferDescriptor {
    explicit BufferDescriptor(const layout& buffer_layout, bool lockable = false, bool shareable = true)
        : m_lockable(lockable),
          m_shareable(shareable),
          m_layout(buffer_layout) {}

    BufferDescriptor(const ov::PartialShape& shape, ov::element::Type type, bool lockable = false, bool shareable = true)
        : BufferDescriptor(layout(shape, type, format::bfyx), lockable, shareable) {}

    BufferDescriptor(size_t elements_count, ov::element::Type type, bool lockable = false, bool shareable = true)
        : BufferDescriptor(layout({static_cast<int64_t>(elements_count)}, type, format::bfyx), lockable, shareable) {}

    bool m_lockable = false;
    bool m_shareable = true;
    layout m_layout;
};

}  // namespace cldnn
