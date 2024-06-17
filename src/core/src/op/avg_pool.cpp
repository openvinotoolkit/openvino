// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/avg_pool.hpp"

#include "avg_pool_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/reference/avg_pool.hpp"

namespace avgpool {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <ov::element::Type_t ET>
    static result_type visit(const ov::Tensor& in,
                             ov::Tensor& out_values,
                             const ov::Shape& in_shape,
                             const ov::Shape& out_shape,
                             const ov::Shape& kernel,
                             const ov::Strides& strides,
                             const ov::Shape& pads_begin,
                             const ov::Shape& pads_end,
                             const bool exclude_pad) {
        using T = typename ov::element_type_traits<ET>::value_type;
        ov::reference::avg_pool<T>(in.data<const T>(),
                                   out_values.data<T>(),
                                   in_shape,
                                   out_shape,
                                   kernel,
                                   strides,
                                   pads_begin,
                                   pads_end,
                                   !exclude_pad);
        return true;
    }
};
}  // namespace avgpool

// ------------------------------ V1 ------------------------------
namespace ov {
namespace op {
namespace v1 {

AvgPool::AvgPool(const Output<Node>& arg,
                 const Strides& strides,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const Shape& kernel,
                 bool exclude_pad,
                 RoundingType rounding_type,
                 const PadType& auto_pad)
    : util::AvgPoolBase(arg, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

void AvgPool::validate_and_infer_types() {
    OV_OP_SCOPE(v1_AvgPool_validate_and_infer_types);

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> AvgPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_AvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<AvgPool>(new_args.at(0),
                                     m_strides,
                                     m_pads_begin,
                                     m_pads_end,
                                     m_kernel,
                                     m_exclude_pad,
                                     m_rounding_type,
                                     m_auto_pad);
}

}  // namespace v1
}  // namespace op
}  // namespace ov

// ------------------------------ V14 ------------------------------
namespace ov {
namespace op {
namespace v14 {
AvgPool::AvgPool(const Output<Node>& arg,
                 const Strides& strides,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const Shape& kernel,
                 bool exclude_pad,
                 RoundingType rounding_type,
                 const PadType& auto_pad)
    : util::AvgPoolBase(arg, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

void AvgPool::validate_and_infer_types() {
    OV_OP_SCOPE(v14_AvgPool_validate_and_infer_types);

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> AvgPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_AvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<AvgPool>(new_args.at(0),
                                     m_strides,
                                     m_pads_begin,
                                     m_pads_end,
                                     m_kernel,
                                     m_exclude_pad,
                                     m_rounding_type,
                                     m_auto_pad);
}

bool AvgPool::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v14_AvgPool_evaluate);
    const auto input_shapes = std::vector<PartialShape>{inputs[0].get_shape()};
    auto pads_begin = m_pads_begin;
    auto pads_end = m_pads_end;
    const auto output_shapes = shape_infer(this, input_shapes, pads_begin, pads_end);

    outputs[0].set_shape(output_shapes[0].get_shape());
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v14_AvgPool_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      avgpool::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      outputs[0].get_shape(),
                                      get_kernel(),
                                      get_strides(),
                                      get_pads_begin(),
                                      get_pads_end(),
                                      get_exclude_pad());
}
bool AvgPool::has_evaluate() const {
    OV_OP_SCOPE(v14_AvgPool_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v14
}  // namespace op
}  // namespace ov
