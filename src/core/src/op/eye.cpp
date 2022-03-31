// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/eye.hpp"

#include "eye_shape_inference.hpp"
#include "ngraph/runtime/reference/eye.hpp"
#include "ngraph/validation_util.hpp"
#include "itt.hpp"

namespace eye {
namespace {
template <ov::element::Type_t ET>
bool evaluate(const ov::HostTensorPtr& out,
              const int64_t diagonal_index) {
    ngraph::runtime::reference::eye(out->get_data_ptr<ET>(), out->get_shape(), diagonal_index);
    return true;
}

bool evaluate_eye(const ov::HostTensorPtr& out,
                  const int64_t diagonal_index) {
    bool rc = true;
    switch (out->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate, i8, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, u8, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, f16, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, bf16, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, i32, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, f32, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, i64, out, diagonal_index);
        default:
            rc = false;
            break;
    }
    return rc;
}
}  // namespace
}  // namespace eye

ov::op::v9::Eye::Eye(const Output<Node>& num_rows,
                     const Output<Node>& num_columns,
                     const Output<Node>& diagonal_index,
                     const Output<Node>& batch_shape,
                     const ov::element::Type& out_type)
    : Op({num_rows, num_columns, diagonal_index, batch_shape}),
      m_output_type(out_type) {
    constructor_validate_and_infer_types();
}

ov::op::v9::Eye::Eye(const Output<Node>& num_rows,
                     const Output<Node>& num_columns,
                     const Output<Node>& diagonal_index,
                     const ov::element::Type& out_type)
    : Op({num_rows, num_columns, diagonal_index}),
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

    if (get_input_size() >= 3) {
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

        input_shapes = {num_rows_pshape, num_columns_pshape, diagonal_index_pshape};
        if (get_input_size() == 4) {
            const auto &batch_shape_et = get_input_element_type(3);
            NODE_VALIDATION_CHECK(this,
                                  batch_shape_et == element::i32 || batch_shape_et == element::i64,
                                  "Type of the 'batch_shape' should be int32 or int64. Got: ",
                                  batch_shape_et);
            const auto &batch_shape_pshape = get_input_partial_shape(3);
            if (batch_shape_pshape.is_static()) {
                const auto &diagonal_index_rank = batch_shape_pshape.rank().get_length();
                NODE_VALIDATION_CHECK(this, diagonal_index_rank == 1, "'batch_shape' value must be a 1D tensor.");
            } else {
                NODE_VALIDATION_CHECK(this,
                                      batch_shape_pshape.rank().is_static(),
                                      "'batch_shape' should have static shape rank");
                NODE_VALIDATION_CHECK(this, batch_shape_pshape.rank() == 1, "'batch_shape' value must be a 1D tensor.");
            }

            input_shapes.push_back(batch_shape_pshape);
        }
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
    } else if (new_args.size() == 3) {
        return std::make_shared<v9::Eye>(new_args[0], new_args[1], new_args[2], m_output_type);
    } else if (new_args.size() == 4) {
        return std::make_shared<v9::Eye>(new_args[0], new_args[1], new_args[2], new_args[3], m_output_type);
    } else {
        throw ov::Exception("Eye has incorrect input number: " + std::to_string(new_args.size()));
    }
}

bool ov::op::v9::Eye::has_evaluate() const {
    NGRAPH_OP_SCOPE(v9_Eye_has_evaluate);
    switch (m_output_type) {
        case ngraph::element::i8:
        case ngraph::element::u8:
        case ngraph::element::f16:
        case ngraph::element::bf16:
        case ngraph::element::i32:
        case ngraph::element::f32:
        case ngraph::element::i64:
            return true;
        default:
            break;
    }
    return false;
}

bool ov::op::v9::Eye::evaluate(const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v9_Eye_evaluate);
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(outputs, 1) &&
                 ngraph::validate_host_tensor_vector(inputs, get_input_size()));

    const auto& num_rows_data = inputs[0];

    int64_t diagonal_index = 0;
    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape::dynamic()};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    if (get_input_size() > 1) {
        const auto& num_columns_data = inputs[1];
        const auto& diagonal_index_data = inputs[2];

        switch (diagonal_index_data->get_element_type()) {
        case element::i32:
            diagonal_index = diagonal_index_data->get_data_ptr<const int32_t>()[0];
            break;
        case element::i64:
            diagonal_index = diagonal_index_data->get_data_ptr<const int64_t>()[0];
            break;
        default:
            throw std::runtime_error("Unsupported type of input `diagonal_index` in EyeLike operation: " +
                                     diagonal_index_data->get_element_type().get_type_name());
        }

        constant_data = {{0, num_rows_data}, {1, num_columns_data}, {2, diagonal_index_data}};

        input_shapes = {num_rows_data->get_partial_shape(),
                        num_columns_data->get_partial_shape(),
                        diagonal_index_data->get_partial_shape()};

        if (get_input_size() > 3) {
            const auto& batch_shape_data = inputs[3];
            constant_data.insert({3, batch_shape_data});
            input_shapes.push_back(batch_shape_data->get_partial_shape());
        }
    } else {
        constant_data = {{0, num_rows_data}};
        input_shapes = {num_rows_data->get_partial_shape()};
    }

    shape_infer(this, input_shapes, output_shapes, constant_data);

    NGRAPH_CHECK(ov::PartialShape(output_shapes[0]).is_static());
    ov::Shape output_shape = output_shapes[0].to_shape();
    outputs[0]->set_element_type(get_out_type());
    outputs[0]->set_shape(output_shape);

    return eye::evaluate_eye(outputs[0], diagonal_index);
}
