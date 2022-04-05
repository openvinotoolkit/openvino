// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "ngraph/runtime/aligned_buffer.hpp"

namespace ngraph {
namespace runtime {
/// \brief SharedBuffer class to store pointer to pre-acclocated buffer.
template <typename T>
class SharedBuffer : public ngraph::runtime::AlignedBuffer {
public:
    SharedBuffer(char* data, size_t size, const T& shared_object) : _shared_object(shared_object) {
        m_allocated_buffer = data;
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    virtual ~SharedBuffer() {
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

private:
    T _shared_object;
};
}  // namespace runtime
}  // namespace ngraph
