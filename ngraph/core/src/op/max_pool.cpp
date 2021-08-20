// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/max_pool.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::MaxPool, "MaxPool", 1, op::util::MaxPoolBase);

op::v1::MaxPool::MaxPool(const Output<Node>& arg,
                         const Strides& strides,
                         const Shape& pads_begin,
                         const Shape& pads_end,
                         const Shape& kernel,
                         const op::RoundingType rounding_type,
                         const PadType auto_pad)
    : op::util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::MaxPool::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v1_MaxPool_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("rounding_type", m_rounding_type);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::v1::MaxPool::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_MaxPool_validate_and_infer_types);

    MaxPoolBase::validate_and_infer_types();

    const PartialShape output_shape = infer_output_shape(Strides{});  // no dilations of the filter window

    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v1::MaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v1_MaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::MaxPool>(new_args.at(0),
                                    m_strides,
                                    m_pads_begin,
                                    m_pads_end,
                                    m_kernel,
                                    m_rounding_type,
                                    m_auto_pad);
}

shared_ptr<Node> op::v1::MaxPool::get_default_value() const {
    return op::Constant::create(get_element_type(), get_shape(), {0});
}

namespace maxpool {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg,
                     const HostTensorPtr& out,
                     const Shape& out_shape,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding_below,
                     const Shape& padding_above) {
    using T = typename element_type_traits<ET>::value_type;
    out->set_shape(out_shape);
    runtime::reference::max_pool<T>(arg->get_data_ptr<ET>(),
                                    out->get_data_ptr<ET>(),
                                    arg->get_shape(),
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above);
    return true;
}

bool evaluate_maxpool(const HostTensorPtr& arg,
                      const HostTensorPtr& out,
                      const Shape& out_shape,
                      const Shape& kernel,
                      const Strides& strides,
                      const Shape& pad_begin,
                      const Shape& pad_end) {
    bool rc = true;
    auto arg_shape = arg->get_shape();

    switch (out->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_maxpool, i32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, i64, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, u32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, u64, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, f16, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
        NGRAPH_TYPE_CASE(evaluate_maxpool, f32, arg, out, out_shape, kernel, strides, pad_begin, pad_end);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace maxpool

bool op::v1::MaxPool::evaluate_maxpool(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    auto arg_shape = inputs[0]->get_partial_shape();
    auto pads_begin_s = get_pads_begin();
    auto pads_end_s = get_pads_end();
    update_auto_padding(arg_shape, Strides(m_kernel.size(), 1), pads_begin_s, pads_end_s);
    CoordinateDiff pads_begin(pads_begin_s.begin(), pads_begin_s.end());
    CoordinateDiff pads_end(pads_end_s.begin(), pads_end_s.end());
    auto out_shape = infer_batched_pooling_forward(this,
                                                   arg_shape,
                                                   pads_begin,
                                                   pads_end,
                                                   get_kernel(),
                                                   get_strides(),
                                                   true,
                                                   get_rounding_type() == op::RoundingType::CEIL,
                                                   Strides{});  // no dilation of the window

    return maxpool::evaluate_maxpool(inputs[0],
                                     outputs[0],
                                     out_shape.get_shape(),
                                     get_kernel(),
                                     get_strides(),
                                     get_pads_begin(),
                                     get_pads_end());
}
bool op::v1::MaxPool::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v1_MaxPool_evaluate);
    return evaluate_maxpool(outputs, inputs);
}

bool op::v1::MaxPool::has_evaluate() const {
    NGRAPH_OP_SCOPE(v1_MaxPool_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

// ------------------------------ V8 ------------------------------

NGRAPH_RTTI_DEFINITION(op::v8::MaxPool, "MaxPool", 8, op::util::MaxPoolBase);

op::v8::MaxPool::MaxPool(const Output<Node>& arg,
                         const Strides& strides,
                         const Strides& dilations,
                         const Shape& pads_begin,
                         const Shape& pads_end,
                         const Shape& kernel,
                         const op::RoundingType rounding_type,
                         const PadType auto_pad,
                         const element::Type index_element_type,
                         const int64_t axis)
    : op::util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad),
      m_dilations{dilations},
      m_index_element_type{index_element_type},
      m_axis{axis} {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v8::MaxPool::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v8_MaxPool_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("rounding_type", m_rounding_type);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("index_element_type", m_index_element_type);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v8::MaxPool::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v8_MaxPool_validate_and_infer_types);

    MaxPoolBase::validate_and_infer_types();

    const auto input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static()) {
        m_axis = ngraph::normalize_axis(this, m_axis, input_shape.rank());
    }

    const PartialShape output_shape = infer_output_shape(m_dilations);

    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, m_index_element_type, output_shape);
}

shared_ptr<Node> op::v8::MaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v8_MaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v8::MaxPool>(new_args.at(0),
                                    m_strides,
                                    m_dilations,
                                    m_pads_begin,
                                    m_pads_end,
                                    m_kernel,
                                    m_rounding_type,
                                    m_auto_pad,
                                    m_index_element_type,
                                    m_axis);
}
