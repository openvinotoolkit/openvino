// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/opsets/opset9.hpp>
#include "utils.hpp"

template<class T>
void shape_infer(const ov::op::v9::Eye* op, const std::vector<T> &input_shapes, std::vector<T> &output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == op->get_input_size() && output_shapes.size() == 1);
    // output_shape = dims_batch_shape + dims_matrix
    std::vector<ov::Dimension> dims_batch_shape;
    std::vector<ov::Dimension> dims_matrix = {ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    auto& dim_num_rows = dims_matrix[0];
    auto& dim_num_columns = dims_matrix[1];
    auto& output_shape = output_shapes[0];

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

    if (op->get_input_size() == 4) {
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

        const auto batch_shape_pshape = input_shapes[3];
        if (batch_shape_pshape.is_static()) {
            dims_batch_shape.resize(batch_shape_pshape[0].get_length(), ov::Dimension::dynamic());
            std::vector<int64_t> batch_shape;
            if (get_data_as_int64<T>(3, op, batch_shape, constant_data)) {
                NODE_VALIDATION_CHECK(op, batch_shape_pshape[0].get_length() == batch_shape.size());
                dims_batch_shape.resize(batch_shape.size());
                for (auto i = 0; i < batch_shape.size(); i++) {
                    NODE_VALIDATION_CHECK(op, batch_shape[i] >= 0,
                                          "'batch_shape' must have non-negative values. Got: ",
                                          batch_shape[i]);

                    dims_batch_shape[i] = batch_shape[i];
                }
            }
        } else {
            output_shape = ov::PartialShape::dynamic();
            return;
        }
    } else {
        dim_num_columns = dim_num_rows;
    }

    std::vector<ov::Dimension> output_dims(std::move(dims_batch_shape));
    output_dims.insert(output_dims.end(), dims_matrix.begin(), dims_matrix.end());
    if (std::any_of(output_dims.begin(), output_dims.end(), [](const ov::Dimension& dim){return dim.is_dynamic(); })) {
        // For PartialShape, Set the output to be dynamic
        // For StaticShape, throw error caused by implicitly constructing StaticShape with PartialShape argument
        output_shape = ov::PartialShape(output_dims);
    } else {
        output_shape = output_dims;
    }
}
