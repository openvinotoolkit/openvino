// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include "col2im_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
<<<<<<< HEAD
#include "openvino/reference/col2im.hpp"
=======
>>>>>>> 239b57869fca7310193d62ad4962673fbe0e4767

namespace ov {
namespace op {
namespace v15 {

Col2Im::Col2Im(const Output<Node>& data,
               const Output<Node>& output_size,
               const Output<Node>& kernel_size,
               const Strides& strides,
               const Strides& dilations,
               const Shape& pads_begin,
               const Shape& pads_end)
    : Op({data, output_size, kernel_size}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end) {
    constructor_validate_and_infer_types();
}

bool Col2Im::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_Col2Im_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    return true;
}

void Col2Im::validate_and_infer_types() {
    OV_OP_SCOPE(v15_Col2Im_validate_and_infer_types);

    const auto& data_element_type = get_input_element_type(0);
    const auto& output_size_element_type = get_input_element_type(1);
    const bool is_valid_output_size_type =
        output_size_element_type == element::i32 || output_size_element_type == element::i64;
    NODE_VALIDATION_CHECK(this,
                          is_valid_output_size_type,
                          "The element type of the output_size tensor must be i32 or i64 type. Got: ",
                          output_size_element_type);

    const auto& kernel_size_element_type = get_input_element_type(2);
    const bool is_valid_kernel_size_type =
        kernel_size_element_type == element::i32 || kernel_size_element_type == element::i64;
    NODE_VALIDATION_CHECK(this,
                          is_valid_kernel_size_type,
                          "The element type of the kernel_size tensor must be i32 or i64 type. Got: ",
                          kernel_size_element_type);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, data_element_type, output_shapes[0]);
}

std::shared_ptr<Node> Col2Im::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v15_Col2Im_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Col2Im>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    m_strides,
                                    m_dilations,
                                    m_pads_begin,
                                    m_pads_end);
}

namespace col2im {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in,
                             const Shape& in_shape,
                             const Tensor& output_size,
                             const Tensor& kernel_size,
                             Tensor& out,
                             const Strides& strides,
                             const Strides& dilations,
                             const Shape& pads_begin,
                             const Shape& pads_end) {
        reference::col2im(in.data<const T>(),
                          in_shape,
                          output_size.data<const T>(),
                          kernel_size.data<const T>(),
                          out.data<T>(),
                          strides,
                          dilations,
                          pads_begin,
                          pads_end);
        return true;
    }
};
}  // namespace col2im

bool Col2Im::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_Col2Im_evaluate);
    const auto output_shape = shape_infer(this, ov::util::get_node_input_partial_shapes(*this)).front();

    outputs[0].set_shape(output_shape.get_shape());
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v15_Col2Im_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      col2im::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1],
                                      inputs[2],
                                      outputs[0],
                                      get_strides(),
                                      get_dilations(),
                                      get_pads_begin(),
                                      get_pads_end());
}

bool Col2Im::has_evaluate() const {
    OV_OP_SCOPE(v1_Col2Im_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}

const Strides& Col2Im::get_strides() const {
    return m_strides;
}

const Strides& Col2Im::get_dilations() const {
    return m_dilations;
}

const Shape& Col2Im::get_pads_begin() const {
    return m_pads_begin;
}

const Shape& Col2Im::get_pads_end() const {
    return m_pads_end;
}

}  // namespace v15
}  // namespace op
}  // namespace ov
