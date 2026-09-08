// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Transpose operator.
 *
 * @param data          Pointer to input data.
 * @param out           Pointer to output data.
 * @param data_shape    Input data shape.
 * @param element_size  Element size in bytes for input and output.
 * @param axes_order    Transpose order.
 * @param out_shape     Output data shape.
 */
void transpose(const char* data,
               char* out,
               const Shape& data_shape,
               size_t element_size,
               const std::vector<int64_t>& axes_order,
               const Shape& out_shape);

/**
 * @brief Reference implementation of Transpose operator for string element type.
 *
 * @param data          Pointer to input string array.
 * @param out           Pointer to output string array.
 * @param data_shape    Input data shape.
 * @param axes_order    Transpose order.
 * @param out_shape     Output data shape.
 */
void transpose(const std::string* data,
               std::string* out,
               const Shape& data_shape,
               const std::vector<int64_t>& axes_order,
               const Shape& out_shape);

/**
 * @brief Reference implementation of Transpose operator for i4/u4 element types.
 *
 * Supports tensors up to 3D. For 3D tensors only the order [0, 2, 1] is supported.
 *
 * @param data          Pointer to input data (packed 4-bit values).
 * @param out           Pointer to output data (packed 4-bit values).
 * @param data_shape    Input data shape.
 * @param axes_order    Transpose order.
 * @param out_shape     Output data shape.
 */
void transpose_4bit(const uint8_t* data,
                    uint8_t* out,
                    const Shape& data_shape,
                    const std::vector<int64_t>& axes_order,
                    const Shape& out_shape);

template <element::Type_t ET>
void transpose_by_iterator(const uint8_t* data,
                           uint8_t* out,
                           const Shape& data_shape,
                           const std::vector<int64_t>& axes_order,
                           const Shape& out_shape) {
    static_assert(ET == element::Type_t::u2 || ET == element::Type_t::u3 || ET == element::Type_t::u6,
                  "transpose_by_iterator supports only u2, u3, and u6 element types");

    const size_t ndim = data_shape.size();
    auto in_it = ov::element::iterator<ET>(reinterpret_cast<const int8_t*>(data));
    auto out_it = ov::element::iterator<ET>(reinterpret_cast<int8_t*>(out));

    ov::Coordinate src_coord(ndim);
    const ov::CoordinateTransformBasic dst_transform{out_shape};
    for (const auto& dst_coord : dst_transform) {
        for (size_t j = 0; j < ndim; ++j)
            src_coord[axes_order[j]] = dst_coord[j];

        const size_t dst_idx = ov::coordinate_index(dst_coord, out_shape);
        const size_t src_idx = ov::coordinate_index(src_coord, data_shape);
        *(out_it + dst_idx) = *(in_it + src_idx);
    }
}

}  // namespace reference
}  // namespace ov
