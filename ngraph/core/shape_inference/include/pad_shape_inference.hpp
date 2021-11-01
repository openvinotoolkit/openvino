// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/pad.hpp>

#include "shape_infer_utils.hpp"
namespace ov {
namespace op {
namespace v1 {

template <class T>
void shape_infer(const Pad* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    constexpr bool is_dynamic_shape = std::is_base_of<ov::PartialShape, T>::value;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 4) && output_shapes.size() == 1);

    auto& output_shape = output_shapes[0];

    // Check the shape of pad_value
    if (op->m_pad_mode == PadMode::CONSTANT && input_shapes.size() == 4) {
        const auto& pad_value_shape = input_shapes[3];
        NODE_VALIDATION_CHECK(op,
                              pad_value_shape.rank().compatible(0),
                              "Argument for padding value is not a scalar (shape: ",
                              pad_value_shape,
                              ").");
    }
    const auto& pads_begin_shape = input_shapes[1];
    const auto& pads_begin_rank = pads_begin_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          pads_begin_rank.compatible(1),
                          "Argument for pads_begin is not 1D (shape: ",
                          pads_begin_rank,
                          ").");

    const auto& pads_end_shape = input_shapes[2];
    const auto& pads_end_rank = pads_end_shape.rank();
    NODE_VALIDATION_CHECK(op,
                          pads_end_rank.compatible(1),
                          "Argument for pads_end is not 1D (shape: ",
                          pads_end_rank,
                          ").");

    const auto& arg_shape = input_shapes[0];
    const auto& arg_shape_rank = arg_shape.rank();

    if (arg_shape_rank.is_static()) {
        if (pads_begin_shape.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  pads_begin_shape[0].get_length() <= arg_shape_rank.get_length(),
                                  "Number of elements of pads_begin must be >= 0 and <= arg rank "
                                  "(pads_begin_shape[0]: ",
                                  pads_begin_shape[0],
                                  ").");
        }
        if (pads_end_shape.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  pads_end_shape[0].get_length() <= arg_shape_rank.get_length(),
                                  "Number of elements of pads_end must be >= 0 and <= arg rank (pads_end_shape[0]: ",
                                  pads_end_shape[0],
                                  ").");
        }

        output_shape.resize(arg_shape_rank.get_length());

        const auto& pads_begin_coord = op->get_pads_begin();
        const auto& pads_end_coord = op->get_pads_end();

        // special check for static shape inference
        NODE_VALIDATION_CHECK(op,
                              is_dynamic_shape || (!pads_begin_coord.empty()),
                              "Cannot determined static output shape when pads_begin is not determined.");

        NODE_VALIDATION_CHECK(op,
                              is_dynamic_shape || (!pads_end_coord.empty()),
                              "Cannot determined static output shape when pads_begin is not determined.");

        if (!pads_begin_coord.empty() && !pads_end_coord.empty()) {
            for (size_t i = 0; i < output_shape.size(); i++) {
                ptrdiff_t begin = i < pads_begin_coord.size() ? pads_begin_coord[i] : 0;
                ptrdiff_t end = i < pads_end_coord.size() ? pads_end_coord[i] : 0;

                if (arg_shape[i].is_static()) {
                    const auto& dim = arg_shape[i].get_length();

                    if (op->m_pad_mode == op::PadMode::EDGE) {
                        NODE_VALIDATION_CHECK(
                            op,
                            begin == 0 || dim > 0,
                            "EDGE padding mode does not allow non-zero pad-begin for zero dimension (",
                            begin,
                            ")");
                        NODE_VALIDATION_CHECK(
                            op,
                            end == 0 || dim > 0,
                            "EDGE padding mode does not allow non-zero pad-begin for zero dimension (",
                            end,
                            ")");
                    }
                    if (op->m_pad_mode == op::PadMode::REFLECT) {
                        NODE_VALIDATION_CHECK(
                            op,
                            begin < dim,
                            "REFLECT padding mode requires pad-begin to be less than dimension, but dimension #",
                            i,
                            " got (pad-begin=",
                            begin,
                            ",dimension=",
                            dim,
                            ")");
                        NODE_VALIDATION_CHECK(
                            op,
                            end < dim,
                            "REFLECT padding mode requires pad-end to be less than dimension, but dimension #",
                            i,
                            " got (pad-end=",
                            end,
                            ", dimension=",
                            dim,
                            ")");
                    }

                    if (op->m_pad_mode == op::PadMode::SYMMETRIC) {
                        NODE_VALIDATION_CHECK(op,
                                              begin <= dim,
                                              "SYMMETRIC padding mode requires pad-begin to be no greater than "
                                              "dimension, but dimension #",
                                              i,
                                              " got (pad-begin=",
                                              begin,
                                              ", dimension=",
                                              dim,
                                              ")");
                        NODE_VALIDATION_CHECK(
                            op,
                            end <= dim,
                            "SYMMETRIC padding mode requires pad-end to be no greater than dimension, but dimension #",
                            i,
                            " got (pad-end=",
                            end,
                            ", dimension=",
                            dim,
                            ")");
                    }

                    output_shape[i] = static_cast<size_t>(begin + dim + end);
                } else {
                    output_shape[i] = arg_shape[i] + (begin + end);
                }
            }
        }
    }
}

}  // namespace v1
}  // namespace op
}  // namespace ov
