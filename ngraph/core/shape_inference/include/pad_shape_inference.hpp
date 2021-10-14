// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/convolution.hpp>

namespace ov {
namespace op {
namespace v1 {

// helper to
template <typename T>
T dynamic_rank_shape() {
    return T{};
}

template <>
PartialShape dynamic_rank_shape() {
    return PartialShape::dynamic();
}

template <class T>
void shape_infer(Pad* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimensionType = typename std::decay<decltype(output_shapes[0][0])>::type;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 4));

    output_shapes.resize(1);
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
    if (arg_shape_rank.is_static() && pads_begin_shape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              pads_begin_shape[0].get_length() <= arg_shape_rank.get_length(),
                              "Number of elements of pads_begin must be >= 0 and <= arg rank "
                              "(pads_begin_shape[0]: ",
                              pads_begin_shape[0],
                              ").");
    }
    if (arg_shape_rank.is_static() && pads_end_shape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              pads_end_shape[0].get_length() <= arg_shape_rank.get_length(),
                              "Number of elements of pads_end must be >= 0 and <= arg rank (pads_end_shape[0]: ",
                              pads_end_shape[0],
                              ").");
    }

    if (arg_shape_rank.is_static()) {
        for (size_t i = 0; i < arg_shape.size(); i++) {
            if (arg_shape[i].is_static()) {
                NODE_VALIDATION_CHECK(op,
                                      op->m_pad_mode != op::PadMode::EDGE || arg_shape[i].get_length() >= 1,
                                      "EDGE padding mode requires an input of dimension of "
                                      "at least 1 at each axis.");
                NODE_VALIDATION_CHECK(op,
                                      op->m_pad_mode != op::PadMode::REFLECT || arg_shape[i].get_length() >= 2,
                                      "REFLECT padding mode requires an input of dimension "
                                      "of at least 2 at each spatial axis.");
            }
        }

        output_shape.resize(arg_shape_rank.get_length());

        const auto& pads_begin_coord = op->get_pads_begin();
        const auto& pads_end_coord = op->get_pads_end();
        if (!pads_begin_coord.empty() && !pads_end_coord.empty()) {
            size_t i;
            for (i = 0; i < pads_begin_coord.size(); i++) {
                if (arg_shape[i].is_static()) {
                    const auto& begin = pads_begin_coord[i];
                    const auto& end = pads_end_coord[i];
                    const auto& dim = arg_shape[i].get_length();

                    if (op->m_pad_mode == op::PadMode::REFLECT) {
                        NODE_VALIDATION_CHECK(op,
                                              begin < dim,
                                              "REFLECT padding mode does not allow pad-begin (",
                                              begin,
                                              ") > dimension (",
                                              dim,
                                              ") - 1.");
                        NODE_VALIDATION_CHECK(op,
                                              end < dim,
                                              "REFLECT padding mode does not allow pad-end (",
                                              end,
                                              ") > dimension (",
                                              dim,
                                              ") - 1");
                    }

                    if (op->m_pad_mode == op::PadMode::SYMMETRIC) {
                        NODE_VALIDATION_CHECK(op,
                                              begin <= dim,
                                              "SYMMETRIC padding mode does not allow pad-begin (",
                                              begin,
                                              ") > dimension (",
                                              dim,
                                              ")");
                        NODE_VALIDATION_CHECK(op,
                                              end <= dim,
                                              "SYMMETRIC padding mode does not allow pad-end (",
                                              end,
                                              ") > dimension (",
                                              dim,
                                              ")");
                    }

                    output_shape[i] = static_cast<size_t>(begin + dim + end);
                }
            }

            for (; i < output_shape.size(); i++) {
                output_shape[i] = arg_shape[i];
            }
        }
    } else {
        // output_shape is dynamic ranked
        output_shape = dynamic_rank_shape<T>();
    }
}

}  // namespace v1
}  // namespace op
}  // namespace ov
