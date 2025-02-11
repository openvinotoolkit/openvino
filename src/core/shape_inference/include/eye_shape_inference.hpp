// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <array>

#include "openvino/op/eye.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <class T>
void check_1D_or_scalar_shape(const ov::op::v9::Eye* op, const T& input_shape, const std::string& name) {
    if (input_shape.is_static()) {
        const auto& num_rows_rank = input_shape.rank().get_length();
        NODE_VALIDATION_CHECK(op, num_rows_rank <= 1, name, " value must be a scalar or 1D tensor.");

        if (num_rows_rank == 1) {
            using TRShape = result_shape_t<T>;
            NODE_VALIDATION_CHECK(op, input_shape.compatible(TRShape{1}), name, " value input should have 1 element.");
        }
    }
}

}  // namespace util

namespace eye {
constexpr std::array<char const*, 4> shape_names{"'num_rows'", "'num_columns'", "'diagonal_index'", "'batch_shape'"};
}

namespace v9 {
/**
 * \brief Eye v9 shape inference compute output shapes.
 *
 * \tparam TShape  Type of shape.
 *
 * \param op             Pointer to Eye operator.
 * \param input_shapes   Input shapes of Eye.
 * \param ta             Tensor accessor to constant data.
 * \return               Vector with output shapes.
 */
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Eye* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    const auto& inputs_count = input_shapes.size();
    NODE_VALIDATION_CHECK(op, (inputs_count == 3 || inputs_count == 4));
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    for (size_t i = 0; i < 3; ++i) {
        util::check_1D_or_scalar_shape(op, input_shapes[i], eye::shape_names[i]);
    }

    if (inputs_count == 4) {
        const auto& batch_shape = input_shapes[3];
        NODE_VALIDATION_CHECK(op, batch_shape.rank().compatible(1), eye::shape_names[3], " input must be a 1D tensor.");
        if (batch_shape.is_static()) {
            if (auto batch_as_shape = get_input_const_data_as_shape<TRShape>(op, 3, ta)) {
                NODE_VALIDATION_CHECK(op,
                                      static_cast<int64_t>(batch_shape[0].get_length()) ==
                                          static_cast<int64_t>(batch_as_shape->rank().get_length()));
                output_shape = std::move(*batch_as_shape);
            } else {
                output_shape = PartialShape::dynamic(batch_shape[0].get_length());
            }
        } else {
            output_shape = PartialShape::dynamic();
            return output_shapes;
        }
    }

    using TDimValue = typename TShape::value_type::value_type;
    constexpr auto get_non_negatives = ov::util::InTypeRange<TDimValue>(0, std::numeric_limits<TDimValue>::max());

    for (size_t i = 0; i < 2; ++i) {
        if (auto eye_dim = get_input_const_data_as_shape<TRShape>(op, i, ta, get_non_negatives)) {
            NODE_VALIDATION_CHECK(op,
                                  eye_dim->size() == 1,
                                  eye::shape_names[i],
                                  " value must be a scalar or 1D tensor. Got: ",
                                  eye_dim->size());
            output_shape.push_back(std::move((*eye_dim)[0]));
        } else {
            output_shape.emplace_back(-1);
        }
    }

    return output_shapes;
}
}  // namespace v9
}  // namespace op
}  // namespace ov
