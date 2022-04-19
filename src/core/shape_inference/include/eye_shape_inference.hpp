// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/opsets/opset9.hpp>
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template<class T>
void check_1D_or_scalar_shape(const ov::op::v9::Eye* op, const T& input_shape, const std::string name) {
    if (input_shape.is_static()) {
        const auto& num_rows_rank = input_shape.rank().get_length();
        NODE_VALIDATION_CHECK(op, num_rows_rank <= 1, name + " value must be a scalar or 1D tensor.");

        if (num_rows_rank == 1) {
            NODE_VALIDATION_CHECK(op,
                                  input_shape.compatible(ov::Shape{1}),
                                  name + " value input should have 1 element.");
        }
    }
}

template<class T>
void check_batch_shape_shape(const ov::op::v9::Eye* op, const T& input_shape) {
    if (input_shape.is_static()) {
        const auto& diagonal_index_rank = input_shape.rank().get_length();
        NODE_VALIDATION_CHECK(op, diagonal_index_rank == 1, "'batch_shape' value must be a 1D tensor.");
    } else {
        NODE_VALIDATION_CHECK(op, input_shape.rank().is_static(), "'batch_shape' should have static shape rank");
        NODE_VALIDATION_CHECK(op, input_shape.rank() == 1, "'batch_shape' value must be a 1D tensor.");
    }
}

}  // namespace util

template<class T>
void shape_infer(const ov::op::v9::Eye* op, const std::vector<T> &input_shapes, std::vector<T> &output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == op->get_input_size() && output_shapes.size() == 1);
    // output_shape = dims_batch_shape + dims_matrix
    T batch_shape;
    T dims_matrix = {ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    auto& dim_num_rows = dims_matrix[0];
    auto& dim_num_columns = dims_matrix[1];
    auto& output_shape = output_shapes[0];

    util::check_1D_or_scalar_shape(op, input_shapes[0], "'num_rows'");
    util::check_1D_or_scalar_shape(op, input_shapes[1], "'num_columns'");
    util::check_1D_or_scalar_shape(op, input_shapes[2], "'diagonal_index'");

    std::vector<int64_t> num_rows;
    if (get_data_as_int64<T>(0, op, num_rows, constant_data)) {
        NODE_VALIDATION_CHECK(op,
                              num_rows.size() == 1,
                              "'num_rows' value must be a scalar or 1D tensor. Got: ",
                              num_rows.size());
        NODE_VALIDATION_CHECK(op,
                              num_rows.front() >= 0,
                              "'num_rows' must be non-negative value. Got: ",
                              num_rows.front());
        dim_num_rows = num_rows.front();
    }

    std::vector<int64_t> num_columns;
    if (get_data_as_int64<T>(1, op, num_columns, constant_data)) {
        NODE_VALIDATION_CHECK(op,
                              num_columns.size() == 1,
                              "'num_columns' value must be a scalar or 1D tensor. Got: ",
                              num_columns.size());
        NODE_VALIDATION_CHECK(op,
                              num_columns.front() >= 0,
                              "'num_columns' must be non-negative value. Got: ",
                              num_columns.front());
        dim_num_columns = num_columns.front();
    }

    if (op->get_input_size() == 4) {
        const auto batch_shape_pshape = input_shapes[3];
        util::check_batch_shape_shape(op, batch_shape_pshape);
        if (batch_shape_pshape.is_static()) {
            if (get_data_as_shape<T>(3, op, batch_shape, constant_data)) {
                NODE_VALIDATION_CHECK(op, batch_shape_pshape[0].get_length() == batch_shape.rank().get_length());
            } else {
                batch_shape = ov::PartialShape::dynamic(batch_shape_pshape[0].get_length());
            }
        } else {
            output_shape = ov::PartialShape::dynamic();
            return;
        }
    }

    output_shape = batch_shape;
    output_shape.insert(output_shape.end(), dims_matrix.begin(), dims_matrix.end());
}

}  // namespace op
}  // namespace ov
