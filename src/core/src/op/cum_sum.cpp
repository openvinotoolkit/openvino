// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/reference/cum_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace cumsum {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0,
                             Tensor& out,
                             const int64_t axis,
                             const bool exclusive,
                             const bool reverse) {
        using T = typename element_type_traits<ET>::value_type;
        reference::cumsum(in0.data<const T>(), axis, out.data<T>(), in0.get_shape(), exclusive, reverse);
        return true;
    }
};

namespace {
bool evaluate(const Node& node,
              TensorVector& outputs,
              const TensorVector& inputs,
              const bool exclusive,
              const bool reverse) {
    const auto& data_shape = inputs[0].get_shape();
    outputs[0].set_shape(data_shape);
    const auto axes = ov::util::try_get_normalized_axis_vector(inputs[1], Rank(data_shape.size()), node);
    OPENVINO_ASSERT(!axes.empty(), "CumSum axis input must not be empty");

    using namespace ov::element;
    return IF_TYPE_OF(CumSum_evaluate,
                      f32,
                      cumsum::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      outputs[0],
                      static_cast<int64_t>(axes.front()),
                      exclusive,
                      reverse);
}
}  // namespace
}  // namespace cumsum

namespace v0 {
CumSum::CumSum(const Output<Node>& arg, const Output<Node>& axis, const bool exclusive, const bool reverse)
    : Op({arg, axis}),
      m_exclusive(exclusive),
      m_reverse(reverse) {
    constructor_validate_and_infer_types();
}

CumSum::CumSum(const Output<Node>& arg, const bool exclusive, const bool reverse)
    : Op({arg, op::v0::Constant::create(element::i32, Shape{}, {0})}),
      m_exclusive(exclusive),
      m_reverse(reverse) {
    constructor_validate_and_infer_types();
}

bool CumSum::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_CumSum_visit_attributes);
    visitor.on_attribute("exclusive", m_exclusive);
    visitor.on_attribute("reverse", m_reverse);
    return true;
}

void CumSum::validate_and_infer_types() {
    OV_OP_SCOPE(v0_CumSum_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));

    const auto& axis_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axis_type == element::i32 || axis_type == element::i64,
                          "axis element type must be either int64_t or int32_t but got (",
                          axis_type,
                          ").");

    // No axis input shape check for backward compatibility
}

std::shared_ptr<Node> CumSum::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_CumSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2)
        return std::make_shared<CumSum>(new_args.at(0), new_args.at(1), m_exclusive, m_reverse);
    else {
        return std::make_shared<CumSum>(new_args.at(0), m_exclusive, m_reverse);
    }
}

bool CumSum::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_CumSum_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    return cumsum::evaluate(*this, outputs, inputs, is_exclusive(), is_reverse());
}

bool CumSum::has_evaluate() const {
    OV_OP_SCOPE(v0_CumSum_has_evaluate);
    const auto& input_0_et = get_input_element_type(0);
    const auto& input_1_et = get_input_element_type(1);
    return input_0_et == element::f32 && (input_1_et == element::i32 || input_1_et == element::i64);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
