// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "utils.hpp"
namespace ov {
namespace op {
namespace pad {
constexpr bool is_inf_padding(const std::pair<int64_t, int64_t>& pad_bounds) {
    return (pad_bounds.first == 0) &&
           (ov::util::dim::is_inf_bound(pad_bounds.second) || ov::util::is_max(pad_bounds.second));
}
}  // namespace pad

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::PadBase* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 || input_shapes.size() == 4);

    const auto& pad_mode = op->get_pad_mode();

    // Check the shape of pad_value
    if (pad_mode == PadMode::CONSTANT && input_shapes.size() == 4) {
        const auto& pad_value_shape = input_shapes[3];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               pad_value_shape.rank().compatible(0),
                               "Argument for padding value is not a scalar.");
    }
    const auto& pads_begin_shape = input_shapes[1];
    const auto& pads_begin_rank = pads_begin_shape.rank();

    NODE_SHAPE_INFER_CHECK(op, input_shapes, pads_begin_rank.compatible(1), "Argument for pads_begin is not 1D.");

    const auto& pads_end_shape = input_shapes[2];
    const auto& pads_end_rank = pads_end_shape.rank();
    NODE_SHAPE_INFER_CHECK(op, input_shapes, pads_end_rank.compatible(1), "Argument for pads_end is not 1D.");

    const auto& arg_shape = input_shapes[0];
    const auto& arg_shape_rank = arg_shape.rank();

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    const auto pads_begin_coord = get_input_bounds<TRShape, int64_t>(op, 1, tensor_accessor);
    const auto pads_end_coord = get_input_bounds<TRShape, int64_t>(op, 2, tensor_accessor);

    if (arg_shape_rank.is_static()) {
        const auto arg_rank_len = arg_shape_rank.get_length();

        if (pads_begin_coord && pads_end_coord) {
            NODE_VALIDATION_CHECK(op,
                                  pads_begin_coord->size() == static_cast<size_t>(arg_rank_len),
                                  "length of pads_begin mismatches with rank of input, expect ",
                                  arg_rank_len,
                                  ", but got ",
                                  pads_begin_coord->size());

            NODE_VALIDATION_CHECK(op,
                                  pads_end_coord->size() == static_cast<size_t>(arg_rank_len),
                                  "length of pads_end mismatches with rank of input, expect ",
                                  arg_rank_len,
                                  ", but got ",
                                  pads_end_coord->size());

            output_shape.reserve(arg_shape.size());
            for (size_t i = 0; i < arg_shape.size(); ++i) {
                const auto& begin = (*pads_begin_coord)[i];
                const auto& end = (*pads_end_coord)[i];

                const auto& begin_lb = std::get<0>(begin);
                const auto& end_lb = std::get<0>(end);

                const auto dim_lb = arg_shape[i].get_min_length();

                if (arg_shape[i].is_static()) {
                    if (begin_lb > 0 || end_lb > 0) {
                        NODE_VALIDATION_CHECK(op,
                                              pad_mode != op::PadMode::EDGE || dim_lb >= 1,
                                              "EDGE padding mode requires an input of dimension of "
                                              "at least 1 at each "
                                              "spatial axis.");
                        NODE_VALIDATION_CHECK(op,
                                              pad_mode != op::PadMode::REFLECT || dim_lb >= 2,
                                              "REFLECT padding mode requires an input of dimension "
                                              "of at least 2 at each "
                                              "spatial axis.");
                    }
                    NODE_SHAPE_INFER_CHECK(
                        op,
                        input_shapes,
                        pad_mode != op::PadMode::REFLECT || (cmp::lt(begin_lb, dim_lb) && cmp::lt(end_lb, dim_lb)),
                        "REFLECT padding mode requires that 'pads_begin[D]' and 'pads_end[D]' "
                        "must be not greater than 'data_shape[D] - 1'.");
                    NODE_SHAPE_INFER_CHECK(
                        op,
                        input_shapes,
                        pad_mode != op::PadMode::SYMMETRIC || (cmp::le(begin_lb, dim_lb) && cmp::le(end_lb, dim_lb)),
                        "SYMMETRIC padding mode requires that 'pads_begin[D]' and 'pads_end[D]' "
                        "must be not greater than 'data_shape[D]'.");
                }

                const auto pad_dim_diff_lb = begin_lb + end_lb;
                const auto pad_dim_diff_ub = begin.second + end.second;
                if ((pad_dim_diff_lb != 0) || (pad_dim_diff_ub != 0)) {
                    using namespace ov::util;
                    const auto lb = dim::padded(dim_lb, pad_dim_diff_lb);
                    if (std::is_same<TRShape, PartialShape>::value) {
                        const auto ub = (pad::is_inf_padding(begin) || pad::is_inf_padding(end))
                                            ? dim::inf_bound
                                            : dim::padded(arg_shape[i].get_max_length(), pad_dim_diff_ub);
                        output_shape.emplace_back(lb, ub);
                    } else {
                        output_shape.emplace_back(lb);
                    }
                } else {
                    output_shape.push_back(arg_shape[i]);
                }
            }
        } else {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                pads_begin_rank.is_dynamic() || cmp::le(pads_begin_shape[0].get_length(), arg_rank_len),
                "Number of elements of pads_begin must be >= 0 and <= arg rank");
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                pads_begin_rank.is_dynamic() || cmp::le(pads_end_shape[0].get_length(), arg_rank_len),
                "Number of elements of pads_end must be >= 0 and <= arg rank");
            output_shape.resize(arg_shape_rank.get_length());
        }
    } else {
        output_shape = PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace op
}  // namespace ov
