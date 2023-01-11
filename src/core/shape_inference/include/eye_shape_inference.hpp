// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/opsets/opset9.hpp>

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
            NODE_VALIDATION_CHECK(op, input_shape.compatible(T{1}), name, " value input should have 1 element.");
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
 * \param constant_data  Map of constant data. Default empty.
 * \return * template <class TShape>
 */
template <class TShape>
std::vector<TShape> shape_infer(const Eye* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    const auto& inputs_count = input_shapes.size();
    NODE_VALIDATION_CHECK(op, (inputs_count == 3 || inputs_count == 4));
    TShape output_shape;

    for (size_t i = 0; i < 3; ++i) {
        util::check_1D_or_scalar_shape(op, input_shapes[i], eye::shape_names[i]);
    }

    if (inputs_count == 4) {
        const auto& batch_shape = input_shapes[3];
        NODE_VALIDATION_CHECK(op, batch_shape.rank().compatible(1), eye::shape_names[3], " input must be a 1D tensor.");
        if (batch_shape.is_static()) {
            if (get_data_as_shape<TShape>(3, op, output_shape, constant_data)) {
                NODE_VALIDATION_CHECK(op, batch_shape[0].get_length() == output_shape.rank().get_length());
            } else {
                output_shape = PartialShape::dynamic(batch_shape[0].get_length());
            }
        } else {
            return {ov::PartialShape::dynamic()};
        }
    }

    using TDimValue = typename TShape::value_type::value_type;
    constexpr auto get_non_negatives = sh_infer::tr::InTypeRange<TDimValue>(0, std::numeric_limits<TDimValue>::max());

    for (size_t i = 0; i < 2; ++i) {
        if (auto eye_dim = get_input_const_data_as_shape<TShape>(op, i, constant_data, get_non_negatives)) {
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

    return {output_shape};
}

template <class TShape>
void shape_infer(const Eye* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = shape_infer(op, input_shapes, constant_data);
}
}  // namespace v9
}  // namespace op
}  // namespace ov
