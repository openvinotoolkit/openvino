// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/split.hpp"

#include <stdio.h>

#include <cstring>
#include <iterator>
#include <numeric>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/reference/slice.hpp"

namespace ov {
namespace reference {
namespace {

// Packed data is a linear bit-stream: bit `pos` lives in byte `pos / 8`, ordered from LSB or MSB within the byte.
class BitStream {
public:
    explicit BitStream(bool lsb_first) : m_lsb_first{lsb_first} {}

    uint8_t mask(size_t pos) const {
        const auto bit = pos % 8;
        return static_cast<uint8_t>(1U << (m_lsb_first ? bit : 7 - bit));
    }

    // Copies `count` bits; destination bits must be zero-initialized.
    void copy(const uint8_t* src, size_t src_pos, uint8_t* dst, size_t dst_pos, size_t count) const {
        const auto copy_bit = [&] {
            if (src[src_pos / 8] & mask(src_pos)) {
                dst[dst_pos / 8] |= mask(dst_pos);
            }
            ++src_pos, ++dst_pos, --count;
        };

        if (src_pos % 8 == dst_pos % 8) {
            while (count > 0 && dst_pos % 8 != 0) {
                copy_bit();
            }
            const auto bytes = count / 8;
            std::memcpy(dst + dst_pos / 8, src + src_pos / 8, bytes);
            src_pos += bytes * 8, dst_pos += bytes * 8, count -= bytes * 8;
        }
        while (count > 0) {
            copy_bit();
        }
    }

private:
    bool m_lsb_first;
};
}  // namespace

void split(const char* data,
           const Shape& data_shape,
           const size_t elem_size,
           const int64_t axis,
           const size_t num_splits,
           char** out_data) {
    const size_t part_length = data_shape.at(axis) / num_splits;

    auto output_shape = data_shape;
    output_shape[axis] = part_length;

    Coordinate lower_bounds(data_shape.size(), 0);
    Coordinate upper_bounds = output_shape;
    auto& lb_at_axis = lower_bounds[axis];
    auto& ub_at_axis = upper_bounds[axis];

    const auto out_last = std::next(out_data, num_splits);
    for (auto out_first = out_data; out_first != out_last; ++out_first) {
        reference::slice(data,
                         *out_first,
                         data_shape,
                         lower_bounds,
                         upper_bounds,
                         Strides(lower_bounds.size(), 1),
                         output_shape,
                         elem_size);
        lb_at_axis += part_length;
        ub_at_axis += part_length;
    }
}

void split_packed(const uint8_t* data,
                  const Shape& data_shape,
                  const element::Type& element_type,
                  const size_t axis,
                  const std::vector<size_t>& split_lengths,
                  uint8_t** out_data) {
    const auto bits = element_type.bitwidth();
    OPENVINO_ASSERT(bits < 8, "Packed Split supports only sub-byte element types, got: ", element_type);
    OPENVINO_ASSERT(axis < data_shape.size(), "Split axis is out of range");
    OPENVINO_ASSERT(std::accumulate(split_lengths.begin(), split_lengths.end(), size_t{0}) == data_shape[axis],
                    "Split lengths must sum up to the data dimension at axis");

    const auto outer = shape_size(data_shape.begin(), data_shape.begin() + axis);
    const auto inner = shape_size(data_shape.begin() + axis + 1, data_shape.end());
    const auto axis_dim = data_shape[axis];
    const BitStream stream{element::is_lsb_packed(element_type)};

    size_t offset = 0;
    for (size_t i = 0; i < split_lengths.size(); ++i) {
        const auto row_bits = split_lengths[i] * inner * bits;
        if (outer * row_bits != 0) {
            std::memset(out_data[i], 0, (outer * row_bits + 7) / 8);
            for (size_t o = 0; o < outer; ++o) {
                stream.copy(data, (o * axis_dim + offset) * inner * bits, out_data[i], o * row_bits, row_bits);
            }
        }
        offset += split_lengths[i];
    }
}
}  // namespace reference
}  // namespace ov
