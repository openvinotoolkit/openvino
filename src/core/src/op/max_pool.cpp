// Copyright (C) 2018-2022 Intel Corporation
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

BWDCMP_RTTI_DEFINITION(op::v1::MaxPool);
BWDCMP_RTTI_DEFINITION(op::v8::MaxPool);

op::v1::MaxPool::MaxPool(const Output<Node>& arg,
                         const Strides& strides,
                         const ov::Shape& pads_begin,
                         const ov::Shape& pads_end,
                         const ov::Shape& kernel,
                         const op::RoundingType rounding_type,
                         const PadType auto_pad)
    : op::util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::MaxPool::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_MaxPool_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("rounding_type", m_rounding_type);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::v1::MaxPool::validate_and_infer_types() {
    OV_OP_SCOPE(v1_MaxPool_validate_and_infer_types);

    MaxPoolBase::validate_and_infer_types();

    const ov::PartialShape output_shape = infer_output_shape(Strides{});  // no dilations of the filter window

    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v1::MaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_MaxPool_clone_with_new_inputs);
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
    return op::v0::Constant::create(get_element_type(), get_shape(), {0});
}

namespace maxpool {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg,
                     const HostTensorPtr& out,
                     const ov::Shape& out_shape,
                     const ov::Shape& window_shape,
                     const Strides& window_movement_strides,
                     const ov::Shape& padding_below,
                     const ov::Shape& padding_above) {
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
                      const ov::Shape& out_shape,
                      const ov::Shape& kernel,
                      const Strides& strides,
                      const ov::Shape& pad_begin,
                      const ov::Shape& pad_end) {
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
}  // namespace
}  // namespace maxpool

bool op::v1::MaxPool::evaluate_maxpool(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    auto arg_shape = inputs[0]->get_partial_shape();
    auto pads_begin_s = get_pads_begin();
    auto pads_end_s = get_pads_end();
    update_auto_padding(arg_shape, Strides(m_kernel.size(), 1), pads_end_s, pads_begin_s);
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
    OV_OP_SCOPE(v1_MaxPool_evaluate);
    return evaluate_maxpool(outputs, inputs);
}

bool op::v1::MaxPool::has_evaluate() const {
    OV_OP_SCOPE(v1_MaxPool_has_evaluate);
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

namespace maxpool_v8 {
namespace {
template <element::Type_t Values, element::Type_t Indices>
inline bool evaluate(const HostTensorPtr& data,
                     const HostTensorPtr& values,
                     const HostTensorPtr& indices,
                     const ov::Shape& out_shape,
                     const ov::Shape& kernel,
                     const Strides& strides,
                     const Strides& dilations,
                     const ov::Shape& pads_begin,
                     const ov::Shape& pads_end,
                     const int64_t axis) {
    using Values_t = typename element_type_traits<Values>::value_type;
    using Indices_t = typename element_type_traits<Indices>::value_type;
    runtime::reference::max_pool<Values_t, Indices_t>(data->get_data_ptr<Values_t>(),
                                                      values->get_data_ptr<Values_t>(),
                                                      indices->get_data_ptr<Indices_t>(),
                                                      data->get_shape(),
                                                      out_shape,
                                                      kernel,
                                                      strides,
                                                      dilations,
                                                      pads_begin,
                                                      pads_end,
                                                      axis);
    return true;
}

bool evaluate_maxpool(const HostTensorPtr& data,
                      const HostTensorPtr& values,
                      const HostTensorPtr& indices,
                      const ov::Shape& out_shape,
                      const ov::Shape& kernel,
                      const Strides& strides,
                      const Strides& dilations,
                      const ov::Shape& pads_begin,
                      const ov::Shape& pads_end,
                      const int64_t axis) {
#define EVAL_MAX_POOL_8(data_et, index_et)            \
    NGRAPH_2_TYPES_CASE(maxpool_v8::evaluate_maxpool, \
                        data_et,                      \
                        index_et,                     \
                        data,                         \
                        values,                       \
                        indices,                      \
                        out_shape,                    \
                        kernel,                       \
                        strides,                      \
                        dilations,                    \
                        pads_begin,                   \
                        pads_end,                     \
                        axis)

    bool rc = true;
    switch (indices->get_element_type()) {
    case element::Type_t::i32: {
        switch (data->get_element_type()) {
            EVAL_MAX_POOL_8(i8, i32);
            EVAL_MAX_POOL_8(i32, i32);
            EVAL_MAX_POOL_8(i64, i32);
            EVAL_MAX_POOL_8(u8, i32);
            EVAL_MAX_POOL_8(u32, i32);
            EVAL_MAX_POOL_8(u64, i32);
            EVAL_MAX_POOL_8(f16, i32);
            EVAL_MAX_POOL_8(f32, i32);
        default:
            rc = false;
            break;
        }
    } break;
    case element::Type_t::i64: {
        switch (data->get_element_type()) {
            EVAL_MAX_POOL_8(i8, i64);
            EVAL_MAX_POOL_8(i32, i64);
            EVAL_MAX_POOL_8(i64, i64);
            EVAL_MAX_POOL_8(u8, i64);
            EVAL_MAX_POOL_8(u32, i64);
            EVAL_MAX_POOL_8(u64, i64);
            EVAL_MAX_POOL_8(f16, i64);
            EVAL_MAX_POOL_8(f32, i64);
        default:
            rc = false;
            break;
        }
    } break;
    default:
        rc = false;
        break;
    }

    return rc;
}
}  // namespace
}  // namespace maxpool_v8

op::v8::MaxPool::MaxPool(const Output<Node>& arg,
                         const Strides& strides,
                         const Strides& dilations,
                         const ov::Shape& pads_begin,
                         const ov::Shape& pads_end,
                         const ov::Shape& kernel,
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
    OV_OP_SCOPE(v8_MaxPool_visit_attributes);
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
    OV_OP_SCOPE(v8_MaxPool_validate_and_infer_types);

    MaxPoolBase::validate_and_infer_types();

    const auto input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static()) {
        m_axis = ngraph::normalize_axis(this, m_axis, input_shape.rank());
    }

    const ov::PartialShape output_shape = infer_output_shape(m_dilations);

    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, m_index_element_type, output_shape);
}

shared_ptr<Node> op::v8::MaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_MaxPool_clone_with_new_inputs);
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

bool op::v8::MaxPool::has_evaluate() const {
    OV_OP_SCOPE(v8_MaxPool_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
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

bool op::v8::MaxPool::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v8_MaxPool_evaluate);

    const auto arg_shape = inputs[0]->get_partial_shape();
    auto pads_begin_s = get_pads_begin();
    auto pads_end_s = get_pads_end();
    update_auto_padding(arg_shape, get_dilations(), pads_end_s, pads_begin_s);
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
                                                   get_dilations());

    return maxpool_v8::evaluate_maxpool(inputs[0],
                                        outputs[0],
                                        outputs[1],
                                        out_shape.get_shape(),
                                        get_kernel(),
                                        get_strides(),
                                        get_dilations(),
                                        get_pads_begin(),
                                        get_pads_end(),
                                        get_axis());
}
