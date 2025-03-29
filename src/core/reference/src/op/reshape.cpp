// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/reshape.hpp"

#include <cstring>
#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/reference/utils/coordinate_range.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
namespace {
static size_t get_threshold() {
    // TODO: find a better way, not hardcoded value
    return (1 << 9) * parallel_get_num_threads();
}

static inline void copy_element(char* out, const char* in, size_t elem_size) {
#define CASE(type)                                                          \
    case sizeof(type):                                                      \
        *reinterpret_cast<type*>(out) = *reinterpret_cast<const type*>(in); \
        break;

    switch (elem_size) {
        CASE(int32_t)
        CASE(int64_t)
        CASE(int16_t)
        CASE(int8_t)
    default:
        std::memcpy(out, in, elem_size);
        break;
    }
#undef CASE
}

void reshape_2D(const char* in,
                char* out,
                const Shape& in_shape,
                const AxisVector& axes_order,
                const Shape& out_shape,
                size_t elem_size) {
    size_t num_elements = shape_size(in_shape);
    if (num_elements <= get_threshold()) {
        for (size_t i = 0; i < out_shape[0]; i++) {
            size_t off = i;
            for (size_t j = 0; j < out_shape[1]; j++) {
                copy_element(out, in + off * elem_size, elem_size);
                out += elem_size;
                off += out_shape[0];
            }
        }
    } else {
        ov::parallel_for2d(out_shape[0], out_shape[1], [in, out, &out_shape, elem_size](size_t i, size_t j) {
            size_t in_off = j * out_shape[0] + i;
            size_t out_off = i * out_shape[1] + j;
            copy_element(out + out_off * elem_size, in + in_off * elem_size, elem_size);
        });
    }
}

static std::vector<size_t> get_strides(size_t rank, size_t elem_size, const AxisVector& order, const Shape& in_shape) {
    std::vector<size_t> rev_order(rank);
    for (size_t i = 0; i < rank; i++) {
        rev_order[order[i]] = i;
    }

    std::vector<size_t> strides(rank);
    strides[rev_order[rank - 1]] = elem_size;
    for (size_t i = rank - 1; i > 0; i--) {
        strides[rev_order[i - 1]] = strides[rev_order[i]] * in_shape[i];
    }

    return strides;
}

void reshape_3D(const char* in,
                char* out,
                const Shape& in_shape,
                const AxisVector& axes_order,
                const Shape& out_shape,
                size_t elem_size) {
    size_t num_elements = shape_size(in_shape);
    if (num_elements <= get_threshold()) {
        const auto strides = get_strides(3, elem_size, axes_order, in_shape);

        size_t off_0 = 0;
        for (size_t i = 0; i < out_shape[0]; i++) {
            size_t off_1 = off_0;
            for (size_t j = 0; j < out_shape[1]; j++) {
                size_t in_off = off_1;
                for (size_t k = 0; k < out_shape[2]; k++) {
                    copy_element(out, in + in_off, elem_size);
                    out += elem_size;
                    in_off += strides[2];
                }
                off_1 += strides[1];
            }
            off_0 += strides[0];
        }
    } else {
        ov::parallel_for3d(out_shape[0],
                           out_shape[1],
                           out_shape[2],
                           [in, out, axes_order, &in_shape, &out_shape, elem_size](size_t i, size_t j, size_t k) {
                               size_t in_indexes[3];
                               in_indexes[axes_order[0]] = i;
                               in_indexes[axes_order[1]] = j;
                               in_indexes[axes_order[2]] = k;
                               size_t in_off =
                                   (in_indexes[0] * in_shape[1] + in_indexes[1]) * in_shape[2] + in_indexes[2];
                               size_t out_off = (i * out_shape[1] + j) * out_shape[2] + k;
                               copy_element(out + out_off * elem_size, in + in_off * elem_size, elem_size);
                           });
    }
}

void reshape_4D(const char* in,
                char* out,
                const Shape& in_shape,
                const AxisVector& axes_order,
                const Shape& out_shape,
                size_t elem_size) {
    size_t num_elements = shape_size(in_shape);
    if (num_elements <= get_threshold()) {
        const auto strides = get_strides(4, elem_size, axes_order, in_shape);

        size_t off_0 = 0;
        for (size_t i = 0; i < out_shape[0]; i++) {
            size_t off_1 = off_0;
            for (size_t j = 0; j < out_shape[1]; j++) {
                size_t off_2 = off_1;
                for (size_t k = 0; k < out_shape[2]; k++) {
                    size_t in_off = off_2;
                    for (size_t l = 0; l < out_shape[3]; l++) {
                        copy_element(out, in + in_off, elem_size);
                        out += elem_size;
                        in_off += strides[3];
                    }
                    off_2 += strides[2];
                }
                off_1 += strides[1];
            }
            off_0 += strides[0];
        }
    } else {
        ov::parallel_for4d(
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            [in, out, axes_order, &in_shape, &out_shape, elem_size](size_t i, size_t j, size_t k, size_t l) {
                size_t in_indexes[4];
                in_indexes[axes_order[0]] = i;
                in_indexes[axes_order[1]] = j;
                in_indexes[axes_order[2]] = k;
                in_indexes[axes_order[3]] = l;
                size_t in_off =
                    ((in_indexes[0] * in_shape[1] + in_indexes[1]) * in_shape[2] + in_indexes[2]) * in_shape[3] +
                    in_indexes[3];
                size_t out_off = ((i * out_shape[1] + j) * out_shape[2] + k) * out_shape[3] + l;
                copy_element(out + out_off * elem_size, in + in_off * elem_size, elem_size);
            });
    }
}

void reshape_5D(const char* in,
                char* out,
                const Shape& in_shape,
                const AxisVector& axes_order,
                const Shape& out_shape,
                size_t elem_size) {
    size_t num_elements = shape_size(in_shape);
    if (num_elements <= get_threshold()) {
        const auto strides = get_strides(5, elem_size, axes_order, in_shape);

        size_t off_0 = 0;
        for (size_t i = 0; i < out_shape[0]; i++) {
            size_t off_1 = off_0;
            for (size_t j = 0; j < out_shape[1]; j++) {
                size_t off_2 = off_1;
                for (size_t k = 0; k < out_shape[2]; k++) {
                    size_t off_3 = off_2;
                    for (size_t l = 0; l < out_shape[3]; l++) {
                        size_t in_off = off_3;
                        for (size_t m = 0; m < out_shape[4]; m++) {
                            copy_element(out, in + in_off, elem_size);
                            out += elem_size;
                            in_off += strides[4];
                        }
                        off_3 += strides[3];
                    }
                    off_2 += strides[2];
                }
                off_1 += strides[1];
            }
            off_0 += strides[0];
        }
    } else {
        ov::parallel_for5d(
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            out_shape[4],
            [in, out, axes_order, &in_shape, &out_shape, elem_size](size_t i, size_t j, size_t k, size_t l, size_t m) {
                size_t in_indexes[5];
                in_indexes[axes_order[0]] = i;
                in_indexes[axes_order[1]] = j;
                in_indexes[axes_order[2]] = k;
                in_indexes[axes_order[3]] = l;
                in_indexes[axes_order[4]] = m;
                size_t in_off =
                    (((in_indexes[0] * in_shape[1] + in_indexes[1]) * in_shape[2] + in_indexes[2]) * in_shape[3] +
                     in_indexes[3]) *
                        in_shape[4] +
                    in_indexes[4];
                size_t out_off = (((i * out_shape[1] + j) * out_shape[2] + k) * out_shape[3] + l) * out_shape[4] + m;
                copy_element(out + out_off * elem_size, in + in_off * elem_size, elem_size);
            });
    }
}

void reshape_6D(const char* in,
                char* out,
                const Shape& in_shape,
                const AxisVector& axes_order,
                const Shape& out_shape,
                size_t elem_size) {
    size_t num_elements = shape_size(in_shape);
    if (num_elements <= get_threshold()) {
        const auto strides = get_strides(6, elem_size, axes_order, in_shape);

        size_t off_0 = 0;
        for (size_t i = 0; i < out_shape[0]; i++) {
            size_t off_1 = off_0;
            for (size_t j = 0; j < out_shape[1]; j++) {
                size_t off_2 = off_1;
                for (size_t k = 0; k < out_shape[2]; k++) {
                    size_t off_3 = off_2;
                    for (size_t l = 0; l < out_shape[3]; l++) {
                        size_t off_4 = off_3;
                        for (size_t m = 0; m < out_shape[4]; m++) {
                            size_t in_off = off_4;
                            for (size_t n = 0; n < out_shape[5]; n++) {
                                copy_element(out, in + in_off, elem_size);
                                out += elem_size;
                                in_off += strides[5];
                            }
                            off_4 += strides[4];
                        }
                        off_3 += strides[3];
                    }
                    off_2 += strides[2];
                }
                off_1 += strides[1];
            }
            off_0 += strides[0];
        }
    } else {
        ov::parallel_for6d(
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            out_shape[4],
            out_shape[5],
            [=, &axes_order, &in_shape, &out_shape](size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
                size_t in_indexes[6];
                in_indexes[axes_order[0]] = i;
                in_indexes[axes_order[1]] = j;
                in_indexes[axes_order[2]] = k;
                in_indexes[axes_order[3]] = l;
                in_indexes[axes_order[4]] = m;
                in_indexes[axes_order[5]] = n;
                size_t in_off =
                    ((((in_indexes[0] * in_shape[1] + in_indexes[1]) * in_shape[2] + in_indexes[2]) * in_shape[3] +
                      in_indexes[3]) *
                         in_shape[4] +
                     in_indexes[4]) *
                        in_shape[5] +
                    in_indexes[5];
                size_t out_off = ((((i * out_shape[1] + j) * out_shape[2] + k) * out_shape[3] + l) * out_shape[4] + m) *
                                     out_shape[5] +
                                 n;
                copy_element(out + out_off * elem_size, in + in_off * elem_size, elem_size);
            });
    }
}

std::vector<size_t> reorder(const std::vector<size_t>& origin, const AxisVector& order) {
    std::vector<size_t> reordered = origin;
    auto out = begin(reordered);
    OPENVINO_ASSERT(origin.size() <= order.size());
    for (size_t i = 0; i < origin.size(); ++i) {
        *out = origin.at(order[i]);
        ++out;
    }
    return reordered;
}

void reshape_ND(const char* in,
                char* out,
                const Shape& in_shape,
                const AxisVector& axes_order,
                const Shape& out_shape,
                size_t elem_size) {
    char* output = out;
    const char* const output_end = out + shape_size(out_shape) * elem_size;
    const auto axis_strides = reorder(row_major_strides(in_shape), axes_order);
    for (const auto& coordinate : CoordinateTransformBasic(reorder(in_shape, axes_order))) {
        if (output >= output_end) {
            break;
        }
        const auto elem_offset = std::inner_product(begin(coordinate), end(coordinate), begin(axis_strides), 0ll);
        const auto input = in + elem_offset * elem_size;
        copy_element(output, input, elem_size);
        output += elem_size;
    }
}

bool no_axis_reordering(const AxisVector& axis_order) {
    auto tmp = axis_order;
    std::sort(begin(tmp), end(tmp));
    tmp.erase(std::unique(begin(tmp), end(tmp)), end(tmp));
    return tmp == axis_order;
}
}  // namespace

void reshape(const char* in,
             char* out,
             const Shape& in_shape,
             const AxisVector& axes_order,
             const Shape& out_shape,
             size_t elem_size) {
    if (no_axis_reordering(axes_order)) {
        std::memcpy(out, in, shape_size(in_shape) * elem_size);
        return;
    }

    if (shape_size(in_shape) == 1) {
        copy_element(out, in, elem_size);
        return;
    }

    switch (in_shape.size()) {
    case 0:
        copy_element(out, in, elem_size);
        break;
    case 1:
        std::memcpy(out, in, in_shape[0] * elem_size);
        break;
    case 2:
        reshape_2D(in, out, in_shape, axes_order, out_shape, elem_size);
        break;
    case 3:
        reshape_3D(in, out, in_shape, axes_order, out_shape, elem_size);
        break;
    case 4:
        reshape_4D(in, out, in_shape, axes_order, out_shape, elem_size);
        break;
    case 5:
        reshape_5D(in, out, in_shape, axes_order, out_shape, elem_size);
        break;
    case 6:
        reshape_6D(in, out, in_shape, axes_order, out_shape, elem_size);
        break;
    default:
        reshape_ND(in, out, in_shape, axes_order, out_shape, elem_size);
        break;
    }
}
}  // namespace reference
}  // namespace ov
