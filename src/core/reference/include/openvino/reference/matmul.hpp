// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "openvino/reference/broadcast.hpp"
#include "openvino/reference/reshape.hpp"

namespace ov {
namespace reference {
namespace details {
template <typename T>
void dot(const T* arg0,
         const T* arg1,
         T* out,
         const Shape& arg0_shape,
         const Shape& arg1_shape,
         const Shape& out_shape) {
    std::fill(out, out + shape_size(out_shape), T{0});
    const size_t arg0_rank = arg0_shape.size();
    const size_t arg1_rank = arg1_shape.size();

    // 2D inputs shapes are interpreted as {I, K} x {K, J}
    // If first input is 1D tensor of shape {K}, it is interpreted as {1, K}
    // If second input is 1D tensor of shape {K}, it is interpreted as {K, 1}
    const size_t I_dim = arg0_rank == 1 ? 1 : arg0_shape[arg0_rank - 2];
    const size_t J_dim = arg1_rank == 1 ? 1 : arg1_shape[arg1_rank - 1];
    const size_t K_dim = arg1_rank == 1 ? arg1_shape[arg1_rank - 1] : arg1_shape[arg1_rank - 2];

    for (size_t i = 0; i < I_dim; ++i) {
        for (size_t k = 0; k < K_dim; ++k) {
            const size_t a_idx = i * K_dim + k;
            for (size_t j = 0; j < J_dim; ++j) {
                const size_t b_idx = k * J_dim + j;
                const size_t out_idx = i * J_dim + j;
                out[out_idx] += arg0[a_idx] * arg1[b_idx];
            }
        }
    }
}

std::vector<size_t> get_transpose_order(const Shape& input_shape);
}  // namespace details
/// \brief Reference kernel for matmul computation.
///
/// \tparam T Type of input and output tensors.
///
/// \param arg0 Pointer to the buffer for left operand input tensor.
/// \param arg1 Pointer to the buffer for right operand input tensor.
/// \param out Pointer to the buffer for output tensor. This must be pre-allocated by
///            the caller, and must be large enough to hold a tensor of the correct
///            shape.
/// \param arg0_shape Shape of arg0.
/// \param arg1_shape Shape of arg1.
/// \param out_shape Shape of out.
/// \param transpose_arg0 Flag to indicate if transpose on arg0.
/// \param transpose_arg1 Flag to indicate if transpose on arg1.
template <typename T>
void matmul(const T* arg0,
            const T* arg1,
            T* out,
            const Shape& arg0_shape,
            const Shape& arg1_shape,
            const Shape& out_shape,
            bool transpose_arg0,
            bool transpose_arg1) {
    // Steps to compute matmul:
    // 1) Check inputs and perform transpose on arg if applicable
    // 2) If ranks of both args are 2D and below (no batch dim),
    //    perform dot and return result; otherwise, continue next
    // 3) Check if auto broadcast is needed on args or transposed args,
    //    and perform broadcast if applicable
    // 4) Perform dot on the args or updated args and return result

    // pointers to updated inputs
    const T* arg0_data = arg0;
    const T* arg1_data = arg1;

    // vectors to hold pontential intermediate transpose,
    // broadcast result
    std::vector<T> arg0_new_data;
    std::vector<T> arg1_new_data;

    // vars for updated inputs shapes
    Shape arg0_shape_tmp = arg0_shape;
    Shape arg1_shape_tmp = arg1_shape;

    size_t arg0_rank = arg0_shape.size();
    size_t arg1_rank = arg1_shape.size();
    const size_t out_rank = out_shape.size();

    // Perform transpose if requested
    if (transpose_arg0 && arg0_rank > 1) {
        std::vector<T> tmp(shape_size(arg0_shape));
        auto axis_vector = details::get_transpose_order(arg0_shape);
        std::swap(arg0_shape_tmp[arg0_rank - 1], arg0_shape_tmp[arg0_rank - 2]);
        reshape(reinterpret_cast<const char*>(arg0_data),
                reinterpret_cast<char*>(tmp.data()),
                arg0_shape,
                axis_vector,
                arg0_shape_tmp,
                sizeof(T));
        arg0_new_data.swap(tmp);
        arg0_data = arg0_new_data.data();
    }

    if (transpose_arg1 && arg1_rank > 1) {
        std::vector<T> tmp(shape_size(arg1_shape));
        auto axis_vector = details::get_transpose_order(arg1_shape);
        std::swap(arg1_shape_tmp[arg1_rank - 1], arg1_shape_tmp[arg1_rank - 2]);
        reshape(reinterpret_cast<const char*>(arg1_data),
                reinterpret_cast<char*>(tmp.data()),
                arg1_shape,
                axis_vector,
                arg1_shape_tmp,
                sizeof(T));
        arg1_new_data.swap(tmp);
        arg1_data = arg1_new_data.data();
    }

    // Inputs are 2D and below, perform dot directly
    if (arg0_rank <= 2 && arg1_rank <= 2) {
        details::dot(arg0_data, arg1_data, out, arg0_shape_tmp, arg1_shape_tmp, out_shape);
        return;
    }

    // Check and perform auto-broadcast if needed
    // If one of the arg is 2D or below, no need to
    // do broadcast on it, just use its value for
    // every batch of dot compuatation later

    if (arg0_rank > 2 && arg1_rank > 2) {
        // Align input batches to the output shape
        Shape arg0_br_target_shape(out_shape.begin(), out_shape.end() - 2);
        Shape arg1_br_target_shape(out_shape.begin(), out_shape.end() - 2);

        arg0_br_target_shape.insert(end(arg0_br_target_shape), end(arg0_shape_tmp) - 2, end(arg0_shape_tmp));
        arg1_br_target_shape.insert(end(arg1_br_target_shape), end(arg1_shape_tmp) - 2, end(arg1_shape_tmp));

        std::vector<size_t> broadcast_axes(out_shape.size() - 2);
        std::iota(broadcast_axes.begin(), broadcast_axes.end(), 0);
        if (!broadcast_axes.empty()) {
            // Usual rules of the broadcasting are applied for batch dimensions.
            // If ranks of input arguments are different,
            // the smaller tensor is unsqueezed from the left side of the shape
            // by necessary number of axes to make both shapes of the same rank.
            // Broadcast all batches (last two dimensions represent matrix),
            // expand dim with value 1 to bigger dim if dimensions are not equal.
            if (arg0_br_target_shape != arg0_shape_tmp) {
                std::vector<T> tmp(shape_size(arg0_br_target_shape));
                broadcast(reinterpret_cast<const char*>(arg0_data),
                          reinterpret_cast<char*>(tmp.data()),
                          arg0_shape_tmp,
                          arg0_br_target_shape,
                          broadcast_axes,
                          sizeof(T));

                arg0_shape_tmp = std::move(arg0_br_target_shape);
                arg0_rank = arg0_shape_tmp.size();
                arg0_new_data.swap(tmp);
                arg0_data = arg0_new_data.data();
            }

            if (arg1_br_target_shape != arg1_shape_tmp) {
                std::vector<T> tmp(shape_size(arg1_br_target_shape));
                broadcast(reinterpret_cast<const char*>(arg1_data),
                          reinterpret_cast<char*>(tmp.data()),
                          arg1_shape_tmp,
                          arg1_br_target_shape,
                          broadcast_axes,
                          sizeof(T));
                arg1_shape_tmp = std::move(arg1_br_target_shape);
                arg1_rank = arg1_shape_tmp.size();
                arg1_new_data.swap(tmp);
                arg1_data = arg1_new_data.data();
            }
        }
    }

    // Perform batched dot
    const Shape dot_arg0_shape =
        (arg0_rank > 2) ? Shape{arg0_shape_tmp[arg0_rank - 2], arg0_shape_tmp[arg0_rank - 1]} : arg0_shape_tmp;
    const Shape dot_arg1_shape =
        (arg1_rank > 2) ? Shape{arg1_shape_tmp[arg1_rank - 2], arg1_shape_tmp[arg1_rank - 1]} : arg1_shape_tmp;
    const Shape dot_output_shape = (out_rank > 2 && arg0_rank > 1 && arg1_rank > 1)
                                       ? Shape{out_shape[out_rank - 2], out_shape[out_rank - 1]}
                                       : Shape{out_shape[out_rank - 1]};

    // Calculate number of batches
    size_t output_batch_size = 1;
    if (out_rank <= 2) {
        // Output is {batch_size, dot_result}, i.e.,
        // arg 0 shape {2}, arg1 shape {3, 2, 1}, output shape {3, 1}
        output_batch_size = out_shape[0];
    } else {
        for (size_t i = 0; i < (out_rank - dot_output_shape.size()); i++) {
            output_batch_size *= out_shape[i];
        }
    }
    const size_t arg0_offset = (arg0_rank > 2) ? shape_size(dot_arg0_shape) : 0;
    const size_t arg1_offset = (arg1_rank > 2) ? shape_size(dot_arg1_shape) : 0;
    const size_t output_offset = shape_size(dot_output_shape);
    for (size_t i = 0; i < output_batch_size; i++) {
        details::dot(arg0_data + i * arg0_offset,
                     arg1_data + i * arg1_offset,
                     out + i * output_offset,
                     dot_arg0_shape,
                     dot_arg1_shape,
                     dot_output_shape);
    }
}
}  // namespace reference
}  // namespace ov
