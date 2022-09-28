// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/util/fft_base.hpp>

#include "openvino/core/axis_vector.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {
namespace rfft_common_validation {
enum class RFFTKind { Forward, Inverse };
template <class T>
void validate_input_rank(const ov::op::util::FFTBase* op,
                         const T& input_shape,
                         const T& axes_shape,
                         size_t input_rank,
                         RFFTKind rfft_kind) {
    const size_t min_rank = (rfft_kind == RFFTKind::Forward) ? 1 : 2;
    NODE_VALIDATION_CHECK(op,
                          input_rank >= min_rank,
                          "The input rank must be greater or equal to ",
                          min_rank,
                          ". Got input rank: ",
                          input_rank);

    if (rfft_kind == RFFTKind::Inverse) {
        NODE_VALIDATION_CHECK(op,
                              input_shape[input_rank - 1].compatible(2),
                              "The last dimension of input data must be 2. Got: ",
                              input_shape[input_rank - 1]);
    }

    if (axes_shape.is_dynamic()) {
        return;
    }

    if (rfft_kind == RFFTKind::Forward) {
        NODE_VALIDATION_CHECK(op,
                              input_rank >= static_cast<int64_t>(axes_shape[0].get_length()),
                              "The input rank must be greater than or equal to the number of RDFT op axes. "
                              "Got input rank: ",
                              input_rank,
                              ", number of axes: ",
                              axes_shape[0].get_length());
    } else {
        NODE_VALIDATION_CHECK(op,
                              input_rank >= static_cast<int64_t>(axes_shape[0].get_length() + 1),
                              "The input rank must be greater than number of IRDFT op axes. Got "
                              "input rank: ",
                              input_rank,
                              ", number of axes: ",
                              axes_shape[0].get_length());
    }
}

template <class T>
void validate_axes(const ov::op::util::FFTBase* op,
                   const T& axes_shape,
                   std::vector<int64_t>& axes,
                   size_t input_rank,
                   bool axes_are_known,
                   RFFTKind rfft_kind) {
    if (axes_shape.rank().is_dynamic() || !axes_are_known) {
        return;
    }

    // IRDFT operation supports negative axes to transform. More precisely, according to
    // the IRDFT operation specification, axes should be integers from -(r - 1) to (r - 2)
    // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis
    // 'r - 1 + a'. The reason is the following: real input tensor of the shape
    // [n_0, ..., n_{r - 1}, 2] is interpreted as a complex tensor with the shape
    // [n_0, ..., n_{r - 1}].
    //
    // But RDFT operation supports negative axes to transform in other sense. More precisely,
    // according to the RDFT operation specification, axes should be integers from -r to (r - 1)
    // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis 'r + a'.
    const int64_t axis_correction = (rfft_kind == RFFTKind::Forward) ? input_rank : (input_rank - 1);
    auto axis_min_value = -static_cast<int64_t>(input_rank);
    auto axis_max_value = static_cast<int64_t>(input_rank) - 1;

    // RDFT op axes can contain the last axis
    if (rfft_kind == RFFTKind::Forward) {
        --axis_min_value;
        ++axis_max_value;
    }

    ov::AxisSet axes_set;
    for (int64_t& axis : axes) {
        NODE_VALIDATION_CHECK(op,
                              axis_min_value < axis && axis < axis_max_value,
                              "(I)RDFT op axis ",
                              axis,
                              " must be in the input rank range (",
                              axis_min_value,
                              ", ",
                              axis_max_value,
                              ").");
        if (axis < 0) {
            axis += axis_correction;
        }
        axes_set.insert(static_cast<size_t>(axis));
    }

    NODE_VALIDATION_CHECK(op, axes.size() == axes_set.size(), "(I)RDFT op axes must be unique.");
}

template <class T>
void validate_signal_size(const ov::op::util::FFTBase* op, const T& axes_shape, const T& signal_size_shape) {
    NODE_VALIDATION_CHECK(op,
                          signal_size_shape.rank().compatible(1),
                          "(I)RDFT op signal size input must be 1D tensor. Got signal: ",
                          signal_size_shape);

    if (axes_shape.is_static() && signal_size_shape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              axes_shape[0].compatible(signal_size_shape[0]),
                              "Sizes of inputs 'axes' and 'signal_size' of (I)RDFT op must be equal. "
                              "Got size of 'axes': ",
                              axes_shape[0],
                              "size of 'signal_size': ",
                              signal_size_shape[0]);
    }
}

template <class T>
void shape_validation(const ov::op::util::FFTBase* op,
                      const std::vector<T>& input_shapes,
                      std::vector<int64_t>& axes,
                      bool axes_are_known,
                      RFFTKind rfft_kind) {
    const auto& input_shape = input_shapes[0];
    const auto& axes_shape = input_shapes[1];

    if (input_shape.rank().is_static()) {
        const auto input_rank = input_shape.size();
        validate_input_rank(op, input_shape, axes_shape, input_rank, rfft_kind);
        validate_axes(op, axes_shape, axes, input_rank, axes_are_known, rfft_kind);
    }

    NODE_VALIDATION_CHECK(op, axes_shape.rank().compatible(1), "(I)RDFT op axes input must be 1D tensor.");

    if (input_shapes.size() == 3) {
        const auto& signal_size_shape = input_shapes[2];
        validate_signal_size(op, axes_shape, signal_size_shape);
    }
}
}  // namespace rfft_common_validation
}  // namespace util
}  // namespace op
}  // namespace ov
