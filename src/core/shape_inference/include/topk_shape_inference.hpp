// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/topk.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <typename T>
void shape_infer(const TopK* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    constexpr bool is_dynamic_shape = std::is_base_of<ov::PartialShape, T>::value;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 && output_shapes.size() == 2));
    const auto& input_shape = input_shapes[0];
    const auto input_rank = input_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          input_rank.is_dynamic() || input_rank.get_length() > 0,
                          "Input rank must be greater than 0.");

    const auto& k_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op, k_shape.rank().compatible(0), "The 'K' input must be a scalar.");

    auto output_shape = input_shape;
    if (input_shape.rank().is_static()) {
        ov::PartialShape k_as_shape;
        auto input_rank = static_cast<int64_t>(input_shape.size());
        auto normalized_axis = ov::normalize_axis(op, op->get_provided_axis(), input_rank, -input_rank, input_rank - 1);
        auto& dim_axis = output_shape[normalized_axis];

        if (!is_dynamic_shape) {
            std::vector<int64_t> k_val;
            NODE_VALIDATION_CHECK(op,
                                  get_data_as_int64<T>(1, op, k_val, constant_data),
                                  "determined k is required to infer static shape");

            NODE_VALIDATION_CHECK(op,
                                  k_val.size() == 1,
                                  "Only one value (scalar) should be provided as the 'K' input to TopK",
                                  " (got ",
                                  k_val.size(),
                                  " elements).");
            dim_axis = k_val[0];
        } else if (ov::evaluate_as_partial_shape(op->input_value(1), k_as_shape)) {
            NODE_VALIDATION_CHECK(op,
                                  k_as_shape.size() == 1,
                                  "Only one value (scalar) should be provided as the 'K' input to TopK",
                                  " (got ",
                                  k_as_shape.size(),
                                  " elements).");
            if (k_as_shape[0].is_static()) {
                NODE_VALIDATION_CHECK(op,
                                      k_as_shape[0].get_max_length() >= 0,
                                      "The value of 'K' must not be a negative number.",
                                      " (got ",
                                      k_as_shape[0].get_max_length(),
                                      ").");
                dim_axis = k_as_shape[0].get_length();
            } else {
                // in this dynamic branch we are sure of dim_axis's type
                const auto in_min = dim_axis.get_min_length();
                const auto in_max = dim_axis.get_max_length();

                const auto k_min = k_as_shape[0].get_min_length();
                const auto k_max = k_as_shape[0].get_max_length();

                const auto lower = std::min<Dimension::value_type>(in_min, k_min);
                const auto upper =
                    in_max < 0 ? Dimension::dynamic().get_max_length() : std::max<Dimension::value_type>(in_max, k_max);
                dim_axis = Dimension(lower, upper);
            }
        } else {
            dim_axis = Dimension(0, dim_axis.get_max_length());
        }
    }

    output_shapes[0] = output_shape;
    output_shapes[1] = output_shape;
}  // namespace
}  // namespace v1
}  // namespace op
}  // namespace ov
