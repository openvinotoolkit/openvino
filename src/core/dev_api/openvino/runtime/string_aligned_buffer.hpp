// Copyright (C) 2018-2025 Intel Corporation
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

    virtual size_t get_num_elements() const {
        return m_num_elements;
    }

    virtual ~StringAlignedBuffer();

private:
    StringAlignedBuffer(const StringAlignedBuffer&) = delete;
    StringAlignedBuffer& operator=(const StringAlignedBuffer&) = delete;

protected:
    size_t m_num_elements{};
};

/// \brief SharedStringAlignedBuffer class to store pointer to shared pre-allocated buffer with std::string objects
/// it must not be responsible for deallocation of std::string objects
class OPENVINO_API SharedStringAlignedBuffer : public ov::StringAlignedBuffer {
public:
    SharedStringAlignedBuffer(char* ptr, size_t size);

    virtual ~SharedStringAlignedBuffer();
};

template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>
    : public DirectValueAccessor<std::shared_ptr<ov::StringAlignedBuffer>> {
public:
    AttributeAdapter(std::shared_ptr<ov::StringAlignedBuffer>& value);

    OPENVINO_RTTI("AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>");

    static std::shared_ptr<ov::StringAlignedBuffer> unpack_string_tensor(const char* packed_string_tensor_ptr,
                                                                         size_t packed_string_tensor_size);
    void get_header(std::shared_ptr<uint8_t>& header, size_t& header_size);
    void get_raw_string_by_index(const char*& raw_string_ptr, size_t& raw_string_size, size_t string_ind);

    ~AttributeAdapter() override;

protected:
    std::shared_ptr<uint8_t> m_header;
    size_t m_header_size;
};

template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<ov::SharedStringAlignedBuffer>>
    : public DirectValueAccessor<std::shared_ptr<ov::SharedStringAlignedBuffer>> {
public:
    AttributeAdapter(std::shared_ptr<ov::SharedStringAlignedBuffer>& value);

    OPENVINO_RTTI("AttributeAdapter<std::shared_ptr<ov::SharedStringAlignedBuffer>");

    void get_header(std::shared_ptr<uint8_t>& header, size_t& header_size);
    void get_raw_string_by_index(const char*& raw_string_ptr, size_t& raw_string_size, size_t string_ind);

    ~AttributeAdapter() override;

protected:
    std::shared_ptr<uint8_t> m_header;
    size_t m_header_size;
};

}  // namespace ov
