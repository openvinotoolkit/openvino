// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/max_pool.hpp"

#include "itt.hpp"
#include "max_pool_shape_inference.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/max_pool.hpp"

namespace ov {
namespace op {
namespace v1 {

MaxPool::MaxPool(const Output<Node>& arg,
                 const Strides& strides,
                 const ov::Shape& pads_begin,
                 const ov::Shape& pads_end,
                 const ov::Shape& kernel,
                 const op::RoundingType rounding_type,
                 const PadType auto_pad)
    : op::util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad) {
    constructor_validate_and_infer_types();
}

bool MaxPool::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_MaxPool_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("rounding_type", m_rounding_type);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void MaxPool::validate_and_infer_types() {
    OV_OP_SCOPE(v1_MaxPool_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shapes = shape_infer(this, get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> MaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_MaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v1::MaxPool>(new_args.at(0),
                                         m_strides,
                                         m_pads_begin,
                                         m_pads_end,
                                         m_kernel,
                                         m_rounding_type,
                                         m_auto_pad);
}

namespace maxpool_v1 {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in,
                             Tensor& out,
                             const ov::Shape& kernel,
                             const Strides& strides,
                             const ov::Shape& pads_begin,
                             const ov::Shape& pads_end) {
        using T = typename element_type_traits<ET>::value_type;
        reference::max_pool(in.data<const T>(),
                            out.data<T>(),
                            in.get_shape(),
                            out.get_shape(),
                            kernel,
                            strides,
                            pads_begin,
                            pads_end);
        return true;
    }
};
}  // namespace maxpool_v1

bool MaxPool::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_MaxPool_evaluate);
    const auto input_shapes = std::vector<PartialShape>{inputs[0].get_shape()};
    auto pads_begin = m_pads_begin;
    auto pads_end = m_pads_end;
    auto output_shape = shape_infer(this, input_shapes, pads_begin, pads_end).front();

    outputs[0].set_shape(output_shape.get_shape());
    using namespace ov::element;
    return IfTypeOf<f16, f32, i32, i64, u32, u64>::apply<maxpool_v1::Evaluate>(inputs[0].get_element_type(),
                                                                               inputs[0],
                                                                               outputs[0],
                                                                               get_kernel(),
                                                                               get_strides(),
                                                                               get_pads_begin(),
                                                                               get_pads_end());
}

bool MaxPool::has_evaluate() const {
    OV_OP_SCOPE(v1_MaxPool_has_evaluate);
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
}  // namespace v1
}  // namespace op
}  // namespace ov

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
    ov::reference::max_pool<Values_t, Indices_t>(data->get_data_ptr<Values_t>(),
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
#define EVAL_MAX_POOL_8(data_et, index_et)              \
    OPENVINO_2_TYPES_CASE(maxpool_v8::evaluate_maxpool, \
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

    const auto input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static()) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        m_axis = ngraph::normalize_axis(this, m_axis, input_shape.rank());
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shapes = shape_infer(this, get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, m_index_element_type, output_shapes[1]);
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

    const auto input_shapes = std::vector<PartialShape>{inputs[0]->get_partial_shape()};
    auto pads_begin = m_pads_begin;
    auto pads_end = m_pads_end;
    auto out_shape = shape_infer(this, input_shapes, pads_begin, pads_end).front();

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
