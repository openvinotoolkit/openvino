// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/eye.hpp"

#include "element_visitor.hpp"
#include "eye_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/reference/eye.hpp"

namespace ov {
namespace op {
namespace eye {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(Tensor& out, const Shape& out_shape, const int64_t diagonal_idx) {
        reference::eye(out.data<T>(), out_shape, diagonal_idx);
        return true;
    }
};
}  // namespace eye

namespace v9 {
Eye::Eye(const Output<Node>& num_rows,
         const Output<Node>& num_columns,
         const Output<Node>& diagonal_index,
         const Output<Node>& batch_shape,
         const ov::element::Type& out_type)
    : Op({num_rows, num_columns, diagonal_index, batch_shape}),
      m_output_type(out_type) {
    constructor_validate_and_infer_types();
}

Eye::Eye(const Output<Node>& num_rows,
         const Output<Node>& num_columns,
         const Output<Node>& diagonal_index,
         const ov::element::Type& out_type)
    : Op({num_rows, num_columns, diagonal_index}),
      m_output_type(out_type) {
    constructor_validate_and_infer_types();
}

void Eye::validate_and_infer_types() {
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

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_out_type(), output_shapes[0]);
}

bool Eye::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_Eye_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<ov::Node> Eye::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v9_Eye_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    switch (new_args.size()) {
    case 3:
        return std::make_shared<Eye>(new_args[0], new_args[1], new_args[2], m_output_type);
    case 4:
        return std::make_shared<Eye>(new_args[0], new_args[1], new_args[2], new_args[3], m_output_type);
    default:
        OPENVINO_THROW("Eye has incorrect input number: ", new_args.size());
    }
}

bool Eye::has_evaluate() const {
    OV_OP_SCOPE(v9_Eye_has_evaluate);
    switch (m_output_type) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
    case element::i8:
    case element::i32:
    case element::i64:
    case element::u8:
        return true;
    default:
        return false;
    }
}

bool Eye::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v9_Eye_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    // Inputs size and shapes checked by shape_infer
    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shape = shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape();

    int64_t diagonal_index;
    const auto& diagonal_tensor = inputs[2];
    switch (diagonal_tensor.get_element_type()) {
    case element::i32:
        diagonal_index = diagonal_tensor.data<const fundamental_type_for<element::i32>>()[0];
        break;
    case element::i64:
        diagonal_index = diagonal_tensor.data<const fundamental_type_for<element::i64>>()[0];
        break;
    default:
        OPENVINO_THROW("Unsupported type of input `diagonal_index` in Eye operation: ",
                       diagonal_tensor.get_element_type().to_string());
    }

    outputs[0].set_shape(output_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v9_Eye_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64, i8, i32, i64, u8),
                                      eye::Evaluate,
                                      outputs[0].get_element_type(),
                                      outputs[0],
                                      output_shape,
                                      diagonal_index);
}
}  // namespace v9
}  // namespace op
}  // namespace ov
