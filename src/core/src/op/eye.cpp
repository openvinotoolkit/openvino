// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/eye.hpp"

#include "eye_shape_inference.hpp"
#include "itt.hpp"

ov::op::v9::Eye::Eye(const Output<Node>& num_rows,
                     const Output<Node>& num_columns,
                     const Output<Node>& diagonal_index,
                     const Output<Node>& batch_shape,
                     const ov::element::Type& out_type)
    : Op({num_rows, num_columns, diagonal_index, batch_shape}),
      m_output_type(out_type) {
    constructor_validate_and_infer_types();
}

ov::op::v9::Eye::Eye(const Output<Node>& num_rows, const ov::element::Type& out_type)
    : Op({num_rows}),
      m_output_type(out_type) {
    constructor_validate_and_infer_types();
}

void ov::op::v9::Eye::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v9_Eye_validate_and_infer_types);
    const auto& num_rows_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          num_rows_et == element::i32 || num_rows_et == element::i64,
                          "Type of the 'num_rows' should be int32 or int64. Got: ",
                          num_rows_et);

    const auto& num_rows_pshape = get_input_partial_shape(0);
    if (num_rows_pshape.is_static()) {
        const auto& num_rows_rank = num_rows_pshape.rank().get_length();
        NODE_VALIDATION_CHECK(this, num_rows_rank <= 1, "'num_rows' value must be a scalar or 1D tensor.");

        if (num_rows_rank == 1) {
            NODE_VALIDATION_CHECK(this, num_rows_pshape.compatible(ov::Shape{1}), "'num_rows' should have 1 element.");
        }
    }

    std::vector<ov::PartialShape> input_shapes;

    if (get_input_size() == 4) {
        const auto& num_columns_et = get_input_element_type(1);
        NODE_VALIDATION_CHECK(this,
                              num_columns_et == element::i32 || num_columns_et == element::i64,
                              "Type of the 'num_columns' should be int32 or int64. Got: ",
                              num_columns_et);

        const auto& num_columns_pshape = get_input_partial_shape(1);
        if (num_columns_pshape.is_static()) {
            const auto& num_columns_rank = num_columns_pshape.rank().get_length();
            NODE_VALIDATION_CHECK(this, num_columns_rank <= 1, "'num_columns' value must be a scalar or 1D tensor.");
            if (num_columns_rank == 1) {
                NODE_VALIDATION_CHECK(this,
                                      num_columns_pshape.compatible(ov::Shape{1}),
                                      "'num_columns' should have 1 element.");
            }
        }

        const auto& diagonal_index_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              diagonal_index_et == element::i32 || diagonal_index_et == element::i64,
                              "Type of the 'diagonal_index' should be int32 or int64. Got: ",
                              diagonal_index_et);
        const auto& diagonal_index_pshape = get_input_partial_shape(2);
        if (diagonal_index_pshape.is_static()) {
            const auto& diagonal_index_rank = diagonal_index_pshape.rank().get_length();
            NODE_VALIDATION_CHECK(this,
                                  diagonal_index_rank <= 1,
                                  "'diagonal_index' value must be a scalar or 1D tensor.");

            if (diagonal_index_rank == 1) {
                NODE_VALIDATION_CHECK(this,
                                      diagonal_index_pshape.compatible(ov::Shape{1}),
                                      "'diagonal_index' should have 1 element.");
            }
        }

        const auto& batch_shape_et = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              batch_shape_et == element::i32 || batch_shape_et == element::i64,
                              "Type of the 'batch_shape' should be int32 or int64. Got: ",
                              batch_shape_et);
        const auto& batch_shape_pshape = get_input_partial_shape(3);
        if (batch_shape_pshape.is_static()) {
            const auto& diagonal_index_rank = batch_shape_pshape.rank().get_length();
            NODE_VALIDATION_CHECK(this, diagonal_index_rank == 1, "'batch_shape' value must be a 1D tensor.");
        } else {
            NODE_VALIDATION_CHECK(this,
                                  batch_shape_pshape.rank().is_static(),
                                  "'batch_shape' should have static shape rank");
            NODE_VALIDATION_CHECK(this, batch_shape_pshape.rank() == 1, "'batch_shape' value must be a 1D tensor.");
        }

        input_shapes = {num_rows_pshape, num_columns_pshape, diagonal_index_pshape, batch_shape_pshape};
    } else {
        input_shapes = {num_rows_pshape};
    }

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape::dynamic()};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_out_type(), output_shapes[0]);
}

bool ov::op::v9::Eye::visit_attributes(ov::AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v9_Eye_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<ov::Node> ov::op::v9::Eye::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v9_Eye_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return std::make_shared<v9::Eye>(new_args[0], m_output_type);
    } else if (new_args.size() == 4) {
        return std::make_shared<v9::Eye>(new_args[0], new_args[1], new_args[2], new_args[3], m_output_type);
    } else {
        throw ov::Exception("Eye has incorrect input number: " + std::to_string(new_args.size()));
    }
}
