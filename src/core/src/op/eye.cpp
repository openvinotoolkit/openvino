// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/eye.hpp"

#include "eye_shape_inference.hpp"
#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/reference/eye.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ov {
namespace op {
namespace eye {
namespace {
template <ov::element::Type_t ET>
bool evaluate(const ngraph::HostTensorPtr& out, const int64_t diagonal_index) {
    ov::reference::eye(out->get_data_ptr<ET>(), out->get_shape(), diagonal_index);
    return true;
}

bool evaluate_eye(const ngraph::HostTensorPtr& out, const int64_t diagonal_index) {
    bool rc = true;
    switch (out->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate, i8, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, u8, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, f16, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, bf16, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, i32, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, f32, out, diagonal_index);
        NGRAPH_TYPE_CASE(evaluate, f64, out, diagonal_index);
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

void ov::op::v9::Eye::validate_and_infer_types() {
    OV_OP_SCOPE(v9_Eye_validate_and_infer_types);

    for (size_t i = 0; i < get_input_size(); ++i) {
        const auto& input_et = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              input_et == element::i32 || input_et == element::i64,
                              "Type of the ",
                              eye::shape_names[i],
                              " should be int32 or int64. Got: ",
                              input_et);
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shape = shape_infer(this, get_node_input_partial_shapes(*this)).front();
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, get_out_type(), output_shape);
}

bool ov::op::v9::Eye::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_Eye_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<ov::Node> ov::op::v9::Eye::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v9_Eye_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<v9::Eye>(new_args[0], new_args[1], new_args[2], m_output_type);
    } else if (new_args.size() == 4) {
        return std::make_shared<v9::Eye>(new_args[0], new_args[1], new_args[2], new_args[3], m_output_type);
    } else {
        OPENVINO_THROW("Eye has incorrect input number: ", new_args.size());
    }
}

bool ov::op::v9::Eye::has_evaluate() const {
    OV_OP_SCOPE(v9_Eye_has_evaluate);
    switch (m_output_type) {
    case ov::element::i8:
    case ov::element::u8:
    case ov::element::f16:
    case ov::element::bf16:
    case ov::element::i32:
    case ov::element::f32:
    case ov::element::i64:
        return true;
    default:
        break;
    }
    return false;
}

bool ov::op::v9::Eye::evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const {
    OV_OP_SCOPE(v9_Eye_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, get_input_size()), "Invalid Eye input TensorVector.");
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1), "Invalid Eye output TensorVector.");
    OPENVINO_SUPPRESS_DEPRECATED_END

    int64_t diagonal_index;

    if (get_input_size() > 1) {
        const auto& diagonal_index_data = inputs[2];

        switch (diagonal_index_data->get_element_type()) {
        case element::i32:
            diagonal_index = diagonal_index_data->get_data_ptr<const int32_t>()[0];
            break;
        case element::i64:
            diagonal_index = diagonal_index_data->get_data_ptr<const int64_t>()[0];
            break;
        default:
            OPENVINO_THROW("Unsupported type of input `diagonal_index` in Eye operation: ",
                           diagonal_index_data->get_element_type().to_string());
        }
    } else {
        diagonal_index = 0;
    }

    std::vector<ov::PartialShape> input_shapes;
    input_shapes.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        input_shapes.push_back(inputs[i]->get_partial_shape());
    }

    const auto output_shape = shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape();

    outputs[0]->set_element_type(get_out_type());
    outputs[0]->set_shape(output_shape);

    return eye::evaluate_eye(outputs[0], diagonal_index);
}
}  // namespace op
}  // namespace ov
