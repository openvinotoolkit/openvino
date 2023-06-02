// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/opt_kernel/reshape.hpp"

#include <algorithm>
#include <cstring>
#include <thread>
#include <future>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/reshape.hpp"

using namespace ngraph;

namespace {

static inline void copy_element(char* out, const char* in, size_t elem_size) {
#define CASE(type) \
    case sizeof(type): \
        *reinterpret_cast<type*>(out) = *reinterpret_cast<const type*>(in); \
        break;

    switch (elem_size) {
        CASE(int32_t)
        CASE(int64_t)
        CASE(int16_t)
        CASE(int8_t)
        default:
            memcpy(out, in, elem_size);
            break;
    }
#undef CASE
}

void reshape_in0(const char* in,
                 char* out,
                 const Shape& in_shape,
                 const AxisVector& in_axis_order,
                 const Shape& out_shape,
                 size_t elem_size) {
    memcpy(out, in, elem_size);
}

void reshape_in1(const char* in,
                 char* out,
                 const Shape& in_shape,
                 const AxisVector& in_axis_order,
                 const Shape& out_shape,
                 size_t elem_size) {
    size_t size[1];
    size_t in_index[1];
    size_t* map_index[1];
    for (size_t i = 0; i < 1; i++) {
        size[i] = in_shape[in_axis_order[i]];
        map_index[in_axis_order[i]] = &in_index[i];
    }
    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0]) {
        memcpy(out, in + *map_index[0] * elem_size, elem_size);
        out += elem_size;
    }
}

void reshape_in2(const char* in,
                 char* out,
                 const Shape& in_shape,
                 const AxisVector& in_axis_order,
                 const Shape& out_shape,
                 size_t elem_size) {
    size_t total_work_size = shape_size(in_shape);
    unsigned int n = std::thread::hardware_concurrency() / 2;
    size_t work_per_thread = std::max<size_t>(total_work_size / n, 1u);

    auto thread_fn = [in, out, elem_size, &in_shape] (size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            size_t off = (in_shape[0] * (i % in_shape[1]) + i / in_shape[1]) * elem_size;
            copy_element(out + off, in + i * elem_size, elem_size);
        }
    };

    std::vector<std::future<void>> futures;

    for (size_t start = 0; start < total_work_size; start += work_per_thread) {
        size_t end = std::min(start + work_per_thread, total_work_size);
        futures.push_back(std::async(thread_fn, start, end));
    }

    for (auto& future : futures) {
        future.wait();
    }
}

void reshape_in3(const char* in,
                 char* out,
                 const Shape& in_shape,
                 const AxisVector& in_axis_order,
                 const Shape& out_shape,
                 size_t elem_size) {
    size_t size[3];
    size_t in_index[3];
    size_t* map_index[3];
    for (size_t i = 0; i < 3; i++) {
        size[i] = in_shape[in_axis_order[i]];
        map_index[in_axis_order[i]] = &in_index[i];
    }
    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0]) {
        for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1]) {
            for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2]) {
                // clang-format off
                    memcpy(out,
                           in + (*map_index[0] * in_shape[1] * in_shape[2] +
                                 *map_index[1] * in_shape[2] +
                                 *map_index[2]) * elem_size,
                           elem_size);
                    out += elem_size;
                // clang-format on
            }
        }
    }
}

void reshape_in4(const char* in,
                 char* out,
                 const Shape& in_shape,
                 const AxisVector& in_axis_order,
                 const Shape& out_shape,
                 size_t elem_size) {
    size_t size[4];
    size_t in_index[4];
    size_t* map_index[4];
    for (size_t i = 0; i < 4; i++) {
        size[i] = in_shape[in_axis_order[i]];
        map_index[in_axis_order[i]] = &in_index[i];
    }
    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0]) {
        for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1]) {
            for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2]) {
                for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3]) {
                    // clang-format off
                        memcpy(out,
                               in + (*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] +
                                     *map_index[1] * in_shape[2] * in_shape[3] +
                                     *map_index[2] * in_shape[3] +
                                     *map_index[3]) * elem_size,
                               elem_size);
                        out += elem_size;
                    // clang-format on
                }
            }
        }
    }
}

void reshape_in5(const char* in,
                 char* out,
                 const Shape& in_shape,
                 const AxisVector& in_axis_order,
                 const Shape& out_shape,
                 size_t elem_size) {
    size_t size[5];
    size_t in_index[5];
    size_t* map_index[5];
    for (size_t i = 0; i < 5; i++) {
        size[i] = in_shape[in_axis_order[i]];
        map_index[in_axis_order[i]] = &in_index[i];
    }
    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0]) {
        for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1]) {
            for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2]) {
                for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3]) {
                    for (in_index[4] = 0; in_index[4] < size[4]; ++in_index[4]) {
                        // clang-format off
                            memcpy(out,
                                   in + (*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] * in_shape[4] +
                                         *map_index[1] * in_shape[2] * in_shape[3] * in_shape[4] +
                                         *map_index[2] * in_shape[3] * in_shape[4] +
                                         *map_index[3] * in_shape[4] +
                                         *map_index[4]) * elem_size,
                                   elem_size);
                            out += elem_size;
                        // clang-format on
                    }
                }
            }
        }
    }
}

void reshape_in6(const char* in,
                 char* out,
                 const Shape& in_shape,
                 const AxisVector& in_axis_order,
                 const Shape& out_shape,
                 size_t elem_size) {
    size_t size[6];
    size_t in_index[6];
    size_t* map_index[6];
    for (size_t i = 0; i < 6; i++) {
        size[i] = in_shape[in_axis_order[i]];
        map_index[in_axis_order[i]] = &in_index[i];
    }
    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0]) {
        for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1]) {
            for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2]) {
                for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3]) {
                    for (in_index[4] = 0; in_index[4] < size[4]; ++in_index[4]) {
                        for (in_index[5] = 0; in_index[5] < size[5]; ++in_index[5]) {
                            // clang-format off
                                memcpy(out,
                                       in + (*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] * in_shape[4] * in_shape[5] +
                                             *map_index[1] * in_shape[2] * in_shape[3] * in_shape[4] * in_shape[5] +
                                             *map_index[2] * in_shape[3] * in_shape[4] * in_shape[5] +
                                             *map_index[3] * in_shape[4] * in_shape[5] +
                                             *map_index[4] * in_shape[5] +
                                             *map_index[5]) * elem_size,
                                       elem_size);
                                out += elem_size;
                            // clang-format on
                        }
                    }
                }
            }
        }
    }
}
bool no_axis_reordering(const AxisVector& axis_order) {
    auto tmp = axis_order;
    std::sort(begin(tmp), end(tmp));
    tmp.erase(std::unique(begin(tmp), end(tmp)), end(tmp));
    return tmp == axis_order;
}
}  // namespace
void runtime::opt_kernel::reshape(const char* in,
                                  char* out,
                                  const Shape& in_shape,
                                  const AxisVector& in_axis_order,
                                  const Shape& out_shape,
                                  size_t elem_size) {
    if (no_axis_reordering(in_axis_order)) {
        std::memcpy(out, in, shape_size(in_shape) * elem_size);
        return;
    }

    switch (in_shape.size()) {
    case 0:
        reshape_in0(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    case 1:
        reshape_in1(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    case 2:
        reshape_in2(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    case 3:
        reshape_in3(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    case 4:
        reshape_in4(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    case 5:
        reshape_in5(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    case 6:
        reshape_in6(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    default:
        reference::reshape(in, out, in_shape, in_axis_order, out_shape, elem_size);
        break;
    }
}
