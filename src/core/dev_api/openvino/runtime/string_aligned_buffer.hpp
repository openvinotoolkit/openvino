// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {

/// \brief StringAlignedBuffer class to store pointer to pre-allocated buffer with std::string objects
/// it is responsible for deallocation of std::string objects that will be stored in the buffer
class OPENVINO_API StringAlignedBuffer : public ov::AlignedBuffer {
public:
    StringAlignedBuffer() = default;

    StringAlignedBuffer(size_t num_elements, size_t byte_size, size_t alignment, bool initialize);

    size_t get_num_elements() const {
        return m_num_elements;
    }

    virtual ~StringAlignedBuffer();

private:
    StringAlignedBuffer(const StringAlignedBuffer&) = delete;
    StringAlignedBuffer& operator=(const StringAlignedBuffer&) = delete;

protected:
    size_t m_num_elements;
};

template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>
    : public DirectValueAccessor<std::shared_ptr<ov::StringAlignedBuffer>> {
public:
    AttributeAdapter(std::shared_ptr<ov::StringAlignedBuffer>& value);

    OPENVINO_RTTI("AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>");
};

}  // namespace ov
