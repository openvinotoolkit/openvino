// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/pad.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v1 {

template <class T>
void shape_infer(const Pad* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    constexpr bool is_dynamic_shape = std::is_base_of<ov::PartialShape, T>::value;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 4) && output_shapes.size() == 1);

    auto& output_shape = output_shapes[0];
    auto pad_mode = op->get_pad_mode();

    // Check the shape of pad_value
    if (pad_mode == PadMode::CONSTANT && input_shapes.size() == 4) {
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

        auto pads_begin_coord = op->get_pads_begin();
        auto pads_end_coord = op->get_pads_end();

        if (pads_begin_coord.empty()) {
            auto it = constant_data.find(1);
            if (it != constant_data.end())
                pads_begin_coord = ov::opset1::Constant(it->second).cast_vector<ptrdiff_t>();
        }

        if (pads_end_coord.empty()) {
            auto it = constant_data.find(2);
            if (it != constant_data.end())
                pads_end_coord = ov::opset1::Constant(it->second).cast_vector<ptrdiff_t>();
        }

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
                    output_shape[i] = static_cast<size_t>(begin + dim + end);

                    if (i > 1) {
                        NODE_VALIDATION_CHECK(op,
                                              pad_mode != op::PadMode::EDGE || arg_shape[i].get_length() >= 1,
                                              "EDGE padding mode requires an input of dimension of "
                                              "at least 1 at each "
                                              "spatial axis.");
                        NODE_VALIDATION_CHECK(op,
                                              pad_mode != op::PadMode::REFLECT || arg_shape[i].get_length() >= 2,
                                              "REFLECT padding mode requires an input of dimension "
                                              "of at least 2 at each "
                                              "spatial axis.");
                    }
                    NODE_VALIDATION_CHECK(op,
                                          pad_mode != op::PadMode::REFLECT || (begin < dim && end < dim),
                                          "REFLECT padding mode requires that 'pads_begin[D]' and 'pads_end[D]' "
                                          "must be not greater than 'data_shape[D] - 1'.");
                    NODE_VALIDATION_CHECK(op,
                                          pad_mode != op::PadMode::SYMMETRIC || (begin <= dim && end <= dim),
                                          "SYMMETRIC padding mode requires that 'pads_begin[D]' and 'pads_end[D]' "
                                          "must be not greater than 'data_shape[D]'.");
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
