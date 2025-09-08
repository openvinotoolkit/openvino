// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/string_aligned_buffer.hpp"

#include <numeric>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace {
void aux_unpack_string_tensor(const char* data, size_t size, std::shared_ptr<ov::StringAlignedBuffer>& string_buffer) {
    // unpack string tensor
    // packed format is the following:
    // <num_string>, <1st string offset>,..., <nth string offset>, <1st string raw format>,..., <nth string raw format>
    // check the format of the input bitstream representing the string tensor
    OPENVINO_ASSERT(size >= 4, "Incorrect packed string tensor format: no batch size in the packed string tensor");
    const int32_t* pindices = reinterpret_cast<const int32_t*>(data);
    int32_t num_strings = pindices[0];
    OPENVINO_ASSERT(int32_t(size) >= 4 + 4 + 4 * num_strings,
                    "Incorrect packed string tensor format: the packed string tensor must contain first "
                    "string offset and end indices");
    const int32_t* begin_ids = pindices + 1;
    const int32_t* end_ids = pindices + 2;
    const char* symbols = reinterpret_cast<const char*>(pindices + 2 + num_strings);

    // allocate StringAlignedBuffer to store unpacked strings in std::string objects
    // SharedBuffer to read byte stream is not applicable because we need unpacked format for strings
    string_buffer = std::make_shared<ov::StringAlignedBuffer>(
        num_strings,
        ov::element::string.size() * num_strings,
        64,  // host alignment used the same as in creation of buffer for Constant
        true);
    std::string* src_strings = static_cast<std::string*>(string_buffer->get_ptr());
    for (int32_t idx = 0; idx < num_strings; ++idx) {
        src_strings[idx] = std::string(symbols + begin_ids[idx], symbols + end_ids[idx]);
    }
}

void aux_get_header(const std::shared_ptr<ov::StringAlignedBuffer>& string_aligned_buffer_ptr,
                    std::shared_ptr<uint8_t>& header,
                    size_t& header_size) {
    OPENVINO_ASSERT(string_aligned_buffer_ptr, "StringAlignedBuffer pointer is nullptr");
    // packed format is the following:
    // <num_string>, <1st string offset>,..., <nth string offset>, <1st string raw format>,..., <nth rawformat>
    auto num_elements = string_aligned_buffer_ptr->get_num_elements();
    auto strings = reinterpret_cast<std::string*>(string_aligned_buffer_ptr->get_ptr());

    // first run over all elements: calculate total memory required to hold all strings
    header_size = sizeof(int32_t) * (1 + 1 + num_elements);
    header = std::shared_ptr<uint8_t>(new uint8_t[header_size], std::default_delete<uint8_t[]>());

    int32_t* pindices = reinterpret_cast<int32_t*>(header.get());
    pindices[0] = int32_t(num_elements);
    pindices[1] = 0;
    pindices += 2;
    size_t current_symbols_pos = 0;

    for (size_t ind = 0; ind < num_elements; ++ind) {
        const auto& str = strings[ind];
        current_symbols_pos += str.size();
        *pindices = int32_t(current_symbols_pos);
        ++pindices;
    }
}

void aux_get_raw_string_by_index(const std::shared_ptr<ov::StringAlignedBuffer>& string_aligned_buffer_ptr,
                                 const char*& raw_string_ptr,
                                 size_t& raw_string_size,
                                 size_t string_ind) {
    OPENVINO_ASSERT(string_aligned_buffer_ptr, "StringAlignedBuffer pointer is nullptr");
    OPENVINO_ASSERT(string_ind < string_aligned_buffer_ptr->get_num_elements(),
                    "Incorrect packed string tensor format: no batch size in the packed string tensor");
    const auto strings = string_aligned_buffer_ptr->get_ptr<const std::string>();
    raw_string_ptr = strings[string_ind].data();
    raw_string_size = strings[string_ind].size();
}
}  // namespace

namespace ov {
StringAlignedBuffer::StringAlignedBuffer(size_t num_elements, size_t byte_size, size_t alignment, bool initialize)
    : AlignedBuffer(byte_size, alignment),
      m_num_elements(num_elements) {
    const auto has_enough_size = (sizeof(std::string) * m_num_elements) <= size();
    OPENVINO_ASSERT(has_enough_size,
                    "Allocated memory of size ",
                    size(),
                    " bytes is not enough to store ",
                    m_num_elements,
                    " std::string objects");
    if (initialize) {
        std::uninitialized_fill_n(get_ptr<std::string>(), m_num_elements, std::string{});
    }
}

StringAlignedBuffer::~StringAlignedBuffer() {
    if (m_allocated_buffer) {
        const auto first = get_ptr<std::string>();
        for_each(first, first + m_num_elements, [](std::string& s) {
            using std::string;
            s.~string();
        });
    }
}

SharedStringAlignedBuffer::SharedStringAlignedBuffer(char* ptr, size_t size) {
    m_allocated_buffer = ptr;
    m_aligned_buffer = ptr;
    m_byte_size = size;
    m_num_elements = size / ov::element::string.size();
}

SharedStringAlignedBuffer::~SharedStringAlignedBuffer() {
    // to prevent deallocation in parent dtors.
    m_allocated_buffer = nullptr;
}

AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::AttributeAdapter(
    std::shared_ptr<ov::StringAlignedBuffer>& value)
    : DirectValueAccessor<std::shared_ptr<ov::StringAlignedBuffer>>(value),
      m_header(nullptr),
      m_header_size(0) {}

std::shared_ptr<ov::StringAlignedBuffer>
AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::unpack_string_tensor(const char* packed_string_tensor_ptr,
                                                                                 size_t packed_string_tensor_size) {
    std::shared_ptr<ov::StringAlignedBuffer> string_aligned_buffer;
    aux_unpack_string_tensor(packed_string_tensor_ptr, packed_string_tensor_size, string_aligned_buffer);
    return string_aligned_buffer;
}

void AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::get_header(std::shared_ptr<uint8_t>& header,
                                                                            size_t& header_size) {
    if (!m_header) {
        aux_get_header(m_ref, m_header, m_header_size);
    }
    header = m_header;
    header_size = m_header_size;
}

void AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::get_raw_string_by_index(const char*& raw_string_ptr,
                                                                                         size_t& raw_string_size,
                                                                                         size_t string_ind) {
    aux_get_raw_string_by_index(m_ref, raw_string_ptr, raw_string_size, string_ind);
}

AttributeAdapter<std::shared_ptr<ov::SharedStringAlignedBuffer>>::AttributeAdapter(
    std::shared_ptr<ov::SharedStringAlignedBuffer>& value)
    : DirectValueAccessor<std::shared_ptr<ov::SharedStringAlignedBuffer>>(value),
      m_header(nullptr),
      m_header_size(0) {}

void AttributeAdapter<std::shared_ptr<ov::SharedStringAlignedBuffer>>::get_header(std::shared_ptr<uint8_t>& header,
                                                                                  size_t& header_size) {
    if (!m_header) {
        aux_get_header(m_ref, m_header, m_header_size);
    }
    header = m_header;
    header_size = m_header_size;
}

void AttributeAdapter<std::shared_ptr<ov::SharedStringAlignedBuffer>>::get_raw_string_by_index(
    const char*& raw_string_ptr,
    size_t& raw_string_size,
    size_t string_ind) {
    aux_get_raw_string_by_index(m_ref, raw_string_ptr, raw_string_size, string_ind);
}

AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::~AttributeAdapter() = default;
AttributeAdapter<std::shared_ptr<SharedStringAlignedBuffer>>::~AttributeAdapter() = default;

}  // namespace ov
