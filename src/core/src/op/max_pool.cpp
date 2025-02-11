// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/max_pool.hpp"

#include "itt.hpp"
#include "max_pool_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/max_pool.hpp"

namespace ov {
namespace op {
namespace pooling {
static int64_t get_normalized_axis(const ov::op::util::MaxPoolBase* op, const int64_t axis) {
    const auto rank = op->get_input_partial_shape(0).rank();
    return rank.is_static() ? ov::util::try_normalize_axis(axis, rank, *op) : axis;
}
}  // namespace pooling
static bool has_evaluate_util(const ov::element::Type& element_type) {
    switch (element_type) {
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

namespace maxpool {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in,
                             Tensor& out_values,
                             Tensor& out_indices,
                             const Shape& in_shape,
                             const Shape& out_shape,
                             const Shape& kernel,
                             const Strides& strides,
                             const Strides& dilations,
                             const Shape& pads_begin,
                             const Shape& pads_end,
                             const int64_t axis) {
        using namespace ov::element;
        return IF_TYPE_OF(maxpool_eval_by_idx_type,
                          OV_PP_ET_LIST(i32, i64),
                          EvalByIdxType,
                          out_indices.get_element_type(),
                          in.data<const T>(),
                          out_values.data<T>(),
                          out_indices,
                          in_shape,
                          out_shape,
                          kernel,
                          strides,
                          dilations,
                          pads_begin,
                          pads_end,
                          axis);
    }

private:
    struct EvalByIdxType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t ET, class T, class I = fundamental_type_for<ET>>
        static result_type visit(const T* in_data,
                                 T* out_values_data,
                                 Tensor& out_indices,
                                 const Shape& in_shape,
                                 const Shape& out_shape,
                                 const Shape& kernel,
                                 const Strides& strides,
                                 const Strides& dilations,
                                 const Shape& pads_begin,
                                 const Shape& pads_end,
                                 const int64_t axis) {
            reference::max_pool(in_data,
                                out_values_data,
                                out_indices.data<I>(),
                                in_shape,
                                out_shape,
                                kernel,
                                strides,
                                dilations,
                                pads_begin,
                                pads_end,
                                axis);
            return true;
        }
    };
};

static bool evaluate_util(const ov::op::util::MaxPoolBase* op,
                          TensorVector& outputs,
                          const TensorVector& inputs,
                          const Strides& dilations,
                          const int64_t axis) {
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(MaxPool_evaluate_util,
                                      op,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8, i32, i64, u8, u32, u64),
                                      Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      outputs[1],
                                      inputs[0].get_shape(),
                                      outputs[0].get_shape(),
                                      op->get_kernel(),
                                      op->get_strides(),
                                      dilations,
                                      op->get_pads_begin(),
                                      op->get_pads_end(),
                                      axis);
}

}  // namespace maxpool

namespace v1 {

MaxPool::MaxPool(const Output<Node>& arg,
                 const Strides& strides,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const Shape& kernel,
                 const RoundingType rounding_type,
                 const PadType auto_pad)
    : util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad) {
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

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
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

namespace maxpool {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in,
                             Tensor& out,
                             const Shape& in_shape,
                             const Shape& out_shape,
                             const Shape& kernel,
                             const Strides& strides,
                             const Shape& pads_begin,
                             const Shape& pads_end) {
        reference::max_pool(in.data<const T>(),
                            out.data<T>(),
                            in_shape,
                            out_shape,
                            kernel,
                            strides,
                            pads_begin,
                            pads_end);
        return true;
    }
};
}  // namespace maxpool

bool MaxPool::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_MaxPool_evaluate);
    const auto input_shapes = std::vector<PartialShape>{inputs[0].get_shape()};
    auto pads_begin = m_pads_begin;
    auto pads_end = m_pads_end;
    const auto output_shapes = shape_infer(this, input_shapes, pads_begin, pads_end);

    outputs[0].set_shape(output_shapes[0].get_shape());
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_MaxPool_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      maxpool::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      outputs[0].get_shape(),
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
namespace ov {
namespace op {
namespace v8 {

MaxPool::MaxPool(const Output<Node>& arg,
                 const Strides& strides,
                 const Strides& dilations,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const Shape& kernel,
                 const RoundingType rounding_type,
                 const PadType auto_pad,
                 const element::Type index_element_type,
                 const int64_t axis)
    : util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad),
      m_dilations{dilations},
      m_index_element_type{index_element_type},
      m_axis{axis} {
    constructor_validate_and_infer_types();
}

bool MaxPool::visit_attributes(AttributeVisitor& visitor) {
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

void MaxPool::validate_and_infer_types() {
    OV_OP_SCOPE(v8_MaxPool_validate_and_infer_types);

    const auto normalized_axis = ov::op::pooling::get_normalized_axis(this, m_axis);
    this->set_axis(normalized_axis);

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, m_index_element_type, output_shapes[1]);
}

std::shared_ptr<Node> MaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_MaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v8::MaxPool>(new_args.at(0),
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

bool MaxPool::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v8_MaxPool_evaluate);

    const auto input_shapes = std::vector<PartialShape>{inputs[0].get_shape()};
    auto pads_begin = m_pads_begin;
    auto pads_end = m_pads_end;
    const auto output_shape = shape_infer(this, input_shapes, pads_begin, pads_end).front();

    outputs[0].set_shape(output_shape.get_shape());
    return ov::op::maxpool::evaluate_util(this, outputs, inputs, get_dilations(), get_axis());
}

bool MaxPool::has_evaluate() const {
    OV_OP_SCOPE(v8_MaxPool_has_evaluate);
    return has_evaluate_util(get_input_element_type(0));
}

/// \return The pooling filter's dilations.
const Strides& MaxPool::get_dilations() const noexcept {
    return m_dilations;
}

void MaxPool::set_dilations(const Strides& dilations) {
    m_dilations = dilations;
}

/// \return The data type of the second output tensor (indices).
element::Type MaxPool::get_index_element_type() const noexcept {
    return m_index_element_type;
}

void MaxPool::set_index_element_type(const element::Type index_element_type) {
    m_index_element_type = index_element_type;
}

/// \return The 'axis' attribute value.
int64_t MaxPool::get_axis() const {
    return m_axis;
}
void MaxPool::set_axis(const int64_t axis) {
    m_axis = axis;
}

}  // namespace v8
}  // namespace op
}  // namespace ov

// ------------------------------ V14 ------------------------------
namespace ov {
namespace op {
namespace v14 {

MaxPool::MaxPool(const Output<Node>& arg,
                 const Strides& strides,
                 const Strides& dilations,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const Shape& kernel,
                 const RoundingType rounding_type,
                 const PadType auto_pad,
                 const element::Type index_element_type,
                 const int64_t axis)
    : util::MaxPoolBase(arg, strides, pads_begin, pads_end, kernel, rounding_type, auto_pad),
      m_dilations{dilations},
      m_index_element_type{index_element_type},
      m_axis{axis} {
    constructor_validate_and_infer_types();
}

bool MaxPool::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_MaxPool_visit_attributes);
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

void MaxPool::validate_and_infer_types() {
    OV_OP_SCOPE(v14_MaxPool_validate_and_infer_types);

    const auto normalized_axis = ov::op::pooling::get_normalized_axis(this, m_axis);
    this->set_axis(normalized_axis);

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, m_index_element_type, output_shapes[1]);
}

std::shared_ptr<Node> MaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_MaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v14::MaxPool>(new_args.at(0),
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

bool MaxPool::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v14_MaxPool_evaluate);
    const auto input_shapes = std::vector<PartialShape>{inputs[0].get_shape()};
    auto pads_begin = m_pads_begin;
    auto pads_end = m_pads_end;
    const auto output_shapes = shape_infer(this, input_shapes, pads_begin, pads_end);

    outputs[0].set_shape(output_shapes[0].get_shape());

    return ov::op::maxpool::evaluate_util(this, outputs, inputs, get_dilations(), get_axis());
}

bool MaxPool::has_evaluate() const {
    OV_OP_SCOPE(v14_MaxPool_has_evaluate);
    return has_evaluate_util(get_input_element_type(0));
}

/// \return The pooling filter's dilations.
const Strides& MaxPool::get_dilations() const noexcept {
    return m_dilations;
}

void MaxPool::set_dilations(const Strides& dilations) {
    m_dilations = dilations;
}

/// \return The data type of the second output tensor (indices).
element::Type MaxPool::get_index_element_type() const noexcept {
    return m_index_element_type;
}

void MaxPool::set_index_element_type(const element::Type index_element_type) {
    m_index_element_type = index_element_type;
}

/// \return The 'axis' attribute value.
int64_t MaxPool::get_axis() const {
    return m_axis;
}
void MaxPool::set_axis(const int64_t axis) {
    m_axis = axis;
}

}  // namespace v14
}  // namespace op
}  // namespace ov
