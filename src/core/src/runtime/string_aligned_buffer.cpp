// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/string_aligned_buffer.hpp"

#include <limits>
#include <numeric>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/common_util.hpp"

namespace {
void aux_unpack_string_tensor(const char* const data,
                              const size_t size,
                              std::shared_ptr<ov::StringAlignedBuffer>& string_buffer) {
    // Packed format is the following:
    // <strings_count>, <1st string offset>,..., <nth string offset>, <1st string raw contents>,..., <nth string raw
    // contents>

    using header_element_t = int32_t;  // Type of a single element in the header (strings_count and offsets)

    static_assert(sizeof(header_element_t) <= sizeof(size_t),
                  "size_t must be at least as wide as header_element_t for safe casting of header values");

    OPENVINO_ASSERT(size >= sizeof(header_element_t),
                    "Incorrect packed string tensor format: no strings count in the packed string tensor");

    const auto header = reinterpret_cast<const header_element_t*>(data);
    const auto strings_count_signed = header[0];
    OPENVINO_ASSERT(strings_count_signed >= 0, "Incorrect packed string tensor format: negative number of strings");

    const auto strings_count = static_cast<size_t>(strings_count_signed);

    constexpr size_t strings_count_elements = 1;  // Header size occupied by strings_count
    constexpr size_t last_end_elements = 1;       // Header size occupied by last end offset

    const size_t header_elems =
        strings_count + (strings_count == 0 ? strings_count_elements : strings_count_elements + last_end_elements);

    constexpr size_t element_size = sizeof(header_element_t);

    size_t header_size = 0;
    const bool is_overflow = ov::util::mul_overflow(header_elems, element_size, header_size);
    OPENVINO_ASSERT(!is_overflow, "Incorrect packed string tensor format: header size overflow detected");

    OPENVINO_ASSERT(header_size <= size, "Incorrect packed string tensor format: header exceeds provided buffer size");

    const auto data_region_size = size - header_size;

    // Allocate StringAlignedBuffer to store unpacked strings in std::string objects
    // SharedBuffer to read byte stream is not applicable because we need unpacked format for strings
    constexpr size_t alignment = 64;   // host alignment used the same as in creation of buffer for Constant
    constexpr bool initialize = true;  // initialize std::string objects to be able to assign to them later
    string_buffer = std::make_shared<ov::StringAlignedBuffer>(strings_count,
                                                              ov::element::string.size() * strings_count,
                                                              alignment,
                                                              initialize);

    std::string* strings = static_cast<std::string*>(string_buffer->get_ptr());

    const header_element_t* begin_offsets = header + 1;
    const header_element_t* end_offsets = header + 2;
    const char* const data_region = reinterpret_cast<const char*>(header + header_elems);

    for (size_t idx = 0; idx < strings_count; ++idx, ++strings, ++begin_offsets, ++end_offsets) {
        const auto begin_signed = *begin_offsets;
        const auto end_signed = *end_offsets;

        OPENVINO_ASSERT(begin_signed >= 0 && end_signed >= 0,
                        "Incorrect packed string tensor format: negative string offset in the packed string tensor");

        OPENVINO_ASSERT(begin_signed <= end_signed,
                        "Incorrect packed string tensor format: begin offset greater than end offset");

        const size_t begin = static_cast<size_t>(begin_signed);
        const size_t end = static_cast<size_t>(end_signed);

        OPENVINO_ASSERT(end <= data_region_size,
                        "Incorrect packed string tensor format: string offset exceeds buffer bounds");

        strings->assign(data_region + begin, data_region + end);
    }
}

void aux_get_header(const std::shared_ptr<ov::StringAlignedBuffer>& string_aligned_buffer_ptr,
                    std::shared_ptr<uint8_t>& data,
                    size_t& header_size) {
    OPENVINO_ASSERT(string_aligned_buffer_ptr, "StringAlignedBuffer pointer is nullptr");
    // Packed format is the following:
    // <strings_count>, <1st string offset>,..., <nth string offset>, <1st string raw contents>,..., <nth string raw
    // contents>
    using header_element_t = int32_t;  // Type of a single element in the header (strings_count and offsets)

    const auto strings_count = string_aligned_buffer_ptr->get_num_elements();

    constexpr size_t strings_count_elements = 1;  // Header size occupied by strings_count
    constexpr size_t last_end_elements = 1;       // Header size occupied by last end offset

    constexpr auto header_elements_max = static_cast<size_t>(std::numeric_limits<header_element_t>::max());

    OPENVINO_ASSERT(strings_count <= header_elements_max - strings_count_elements - last_end_elements,
                    "Incorrect StringAlignedBuffer format: header element count overflow");

    const size_t header_elems =
        strings_count + (strings_count == 0 ? strings_count_elements : strings_count_elements + last_end_elements);

    constexpr size_t element_size = sizeof(header_element_t);

    size_t header_size_bytes = 0;
    const bool is_overflow = ov::util::mul_overflow(header_elems, element_size, header_size_bytes);
    OPENVINO_ASSERT(!is_overflow, "Incorrect StringAlignedBuffer format: header size overflow detected");

    header_size = header_size_bytes;
    data = std::shared_ptr<uint8_t>(new uint8_t[header_size], std::default_delete<uint8_t[]>());

    header_element_t* header = reinterpret_cast<header_element_t*>(data.get());
    header[0] = static_cast<header_element_t>(strings_count);

    if (strings_count > 0) {
        header[1] = 0;
        header += 2;
        size_t current_string_end = 0;

        auto strings = reinterpret_cast<std::string*>(string_aligned_buffer_ptr->get_ptr());

        for (size_t idx = 0; idx < strings_count; ++idx, ++header, ++strings) {
            current_string_end += strings->size();
            OPENVINO_ASSERT(
                current_string_end <= header_elements_max,
                "Incorrect StringAlignedBuffer format: total string data size exceeds header element type capacity");
            *header = static_cast<header_element_t>(current_string_end);
        }
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
