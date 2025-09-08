// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/core/axis_vector.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/fft_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {
namespace fft_common_validation {
enum FFTKind { RealInput = 1, ComplexInput = 2 };
template <class T>
void validate_input_rank(const ov::op::util::FFTBase* op,
                         const std::vector<T>& input_shapes,
                         const T& input_shape,
                         const T& axes_shape,
                         int64_t input_rank,
                         FFTKind fft_kind) {
    const int64_t min_rank = fft_kind;
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_rank >= min_rank,
                           "The input rank must be greater or equal to ",
                           min_rank);

    if (fft_kind == FFTKind::ComplexInput) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shape[input_rank - 1].compatible(2),
                               "The last dimension of input data must be 2.");
    }

    if (axes_shape.is_dynamic()) {
        return;
    }

    if (fft_kind == FFTKind::RealInput) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               ov::cmp::ge(input_rank, axes_shape[0].get_length()),
                               "The input rank must be greater than or equal to the number of axes. ");
    } else {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               ov::cmp::ge(input_rank, axes_shape[0].get_length() + 1),
                               "The input rank must be greater than number of axes.");
    }
}

template <class T>
void validate_axes(const ov::op::util::FFTBase* op,
                   const std::vector<T>& input_shapes,
                   const T& axes_shape,
                   std::vector<int64_t>& axes,
                   int64_t input_rank,
                   FFTKind fft_kind) {
    if (axes_shape.rank().is_dynamic()) {
        return;
    }

    // IRDFT, DFT, IDFT, operations supports negative axes to transform. More precisely, according to
    // the operation specification, axes should be integers from -(r - 1) to (r - 2)
    // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis
    // 'r - 1 + a'. The reason is the following: real input tensor of the shape
    // [n_0, ..., n_{r - 1}, 2] is interpreted as a complex tensor with the shape
    // [n_0, ..., n_{r - 1}].
    //
    // But RDFT operation supports negative axes to transform in other sense. More precisely,
    // according to the RDFT operation specification, axes should be integers from -r to (r - 1)
    // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis 'r + a'.
    const int64_t axis_correction = (fft_kind == FFTKind::RealInput) ? input_rank : (input_rank - 1);
    ov::util::try_normalize_axes(axes, axis_correction, *op);
    NODE_VALIDATION_CHECK(op, ov::util::are_unique(axes), "Each axis must be unique.");
}

template <class T>
void validate_signal_size(const ov::op::util::FFTBase* op,
                          const std::vector<T>& input_shapes,
                          const T& axes_shape,
                          const T& signal_size_shape) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           signal_size_shape.rank().compatible(1),
                           "Signal size input must be 1D tensor.");

    if (axes_shape.is_static() && signal_size_shape.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               axes_shape[0].compatible(signal_size_shape[0]),
                               "Sizes of inputs 'axes' and 'signal_size' must be equal.");
    }
}

template <class T>
void shape_validation(const ov::op::util::FFTBase* op,
                      const std::vector<T>& input_shapes,
                      std::optional<std::vector<int64_t>>& axes,
                      FFTKind fft_kind) {
    const auto& input_shape = input_shapes[0];
    const auto& axes_shape = input_shapes[1];

    const auto input_shape_rank = input_shape.rank();
    if (input_shape_rank.is_static()) {
        const auto input_rank_length = input_shape_rank.get_length();
        validate_input_rank(op, input_shapes, input_shape, axes_shape, input_rank_length, fft_kind);
        if (axes) {
            validate_axes(op, input_shapes, axes_shape, *axes, input_rank_length, fft_kind);
        }
    }

    NODE_SHAPE_INFER_CHECK(op, input_shapes, axes_shape.rank().compatible(1), "Axes input must be 1D tensor.");

    if (input_shapes.size() == 3) {
        const auto& signal_size_shape = input_shapes[2];
        validate_signal_size(op, input_shapes, axes_shape, signal_size_shape);
    }
}
}  // namespace fft_common_validation
}  // namespace util
}  // namespace op
}  // namespace ov
