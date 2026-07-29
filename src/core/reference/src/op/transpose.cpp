// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/transpose.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/reshape.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
void transpose(const char* data,
               char* out,
               const Shape& data_shape,
               size_t element_size,
               const std::vector<int64_t>& axes_order,
               const Shape& out_shape) {
    // To reuse reference::reshape axes order vector has to be converted to AxisVector
    // Negative axes are not supported, it is validated by transpose evaluate method
    const AxisVector axes_vector(axes_order.begin(), axes_order.end());
    reshape(data, out, data_shape, axes_vector, out_shape, element_size);
}

void transpose(const std::string* data,
               std::string* out,
               const Shape& data_shape,
               const std::vector<int64_t>& axes_order,
               const Shape& out_shape) {
    const size_t ndim = data_shape.size();
    ov::Coordinate src_coord(ndim);
    const ov::CoordinateTransformBasic dst_transform{out_shape};
    for (const auto& dst_coord : dst_transform) {
        for (size_t j = 0; j < ndim; ++j)
            src_coord[axes_order[j]] = dst_coord[j];
        out[ov::coordinate_index(dst_coord, out_shape)] = data[ov::coordinate_index(src_coord, data_shape)];
    }
}

namespace {
enum class int4_extract_t : uint8_t { low_half = 0, high_half = 4 };

struct int4_iterator {
    explicit int4_iterator(uint8_t* ptr) : m_ptr(ptr), m_half(int4_extract_t::low_half) {}
    explicit int4_iterator(uint8_t* ptr, int4_extract_t half) : m_ptr(ptr), m_half(half) {}

    void operator++() {
        if (m_half == int4_extract_t::low_half) {
            m_half = int4_extract_t::high_half;
        } else {
            m_half = int4_extract_t::low_half;
            m_ptr += 1;
        }
    }

    int4_iterator operator+(const size_t shift) const {
        return int4_iterator{m_ptr + shift / 2, shift % 2 ? int4_extract_t::high_half : int4_extract_t::low_half};
    }

    void copy_from(const int4_iterator& from) const {
        uint8_t from_val = *from.m_ptr;
        uint8_t mask_from = from.m_half == int4_extract_t::high_half ? 0xF0 : 0x0F;
        uint8_t mask_to = m_half == int4_extract_t::high_half ? 0x0F : 0xF0;

        if (from.m_half < m_half) {
            from_val <<= 4;
        } else if (from.m_half > m_half) {
            from_val >>= 4;
        } else {
            from_val &= mask_from;
        }

        *m_ptr = (*m_ptr & mask_to) | from_val;
    }

    uint8_t* m_ptr;
    int4_extract_t m_half;
};

void transpose_xy_4bit(int4_iterator& out_ptr, int4_iterator& in_ptr, size_t out_shape_d0, size_t out_shape_d1) {
    for (size_t i = 0; i < out_shape_d0; i++) {
        size_t off = i;
        for (size_t j = 0; j < out_shape_d1; j++) {
            out_ptr.copy_from(in_ptr + off);
            ++out_ptr;
            off += out_shape_d0;
        }
    }
}
}  // namespace

void transpose_4bit(const uint8_t* data,
                    uint8_t* out,
                    const Shape& data_shape,
                    const std::vector<int64_t>& axes_order,
                    const Shape& out_shape) {
    if (data_shape.size() == 2) {
        auto out_ptr = int4_iterator(out);
        auto in_ptr = int4_iterator(const_cast<uint8_t*>(data));
        transpose_xy_4bit(out_ptr, in_ptr, out_shape[0], out_shape[1]);
    } else if (data_shape.size() == 3) {
        OPENVINO_ASSERT(axes_order[0] == 0 && axes_order[1] == 2 && axes_order[2] == 1,
                        "Unsupported transpose order for i4/u4 type");
        const auto out_batch = out_shape[0];
        const auto out_shape_d0 = out_shape[1];
        const auto out_shape_d1 = out_shape[2];
        OPENVINO_ASSERT((out_shape_d0 * out_shape_d1) % 2 == 0,
                        "Only supports even number of i4/u4 data in each batch for transposing");
        size_t batch_offset = 0;
        for (size_t b = 0; b < out_batch; b++) {
            uint8_t* out_batch_base = out + batch_offset / 2;
            uint8_t* in_batch_base = const_cast<uint8_t*>(data) + batch_offset / 2;
            auto out_ptr = int4_iterator(out_batch_base);
            auto in_ptr = int4_iterator(in_batch_base);
            transpose_xy_4bit(out_ptr, in_ptr, out_shape_d0, out_shape_d1);
            batch_offset += out_shape_d0 * out_shape_d1;
        }
    } else {
        OPENVINO_THROW("Transpose for i4/u4 dtype is supported only for ndims <= 3");
    }
}

}  // namespace reference
}  // namespace ov
