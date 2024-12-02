// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shape_of.hpp"

#include <algorithm>
#include <vector>

#include "itt.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/reference/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace shape_of {
namespace {
template <element::Type_t ET>
inline bool evaluate(const Shape& shape, Tensor& output_value) {
    reference::shape_of(shape, output_value.data<fundamental_type_for<ET>>());
    return true;
}

bool evaluate_shape_of(Tensor& output_value, const Shape& input_shape) {
    bool rc;
    output_value.set_shape(Shape{input_shape.size()});
    switch (output_value.get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_shape_of, i32, input_shape, output_value);
        OPENVINO_TYPE_CASE(evaluate_shape_of, i64, input_shape, output_value);
        OPENVINO_TYPE_CASE(evaluate_shape_of, u32, input_shape, output_value);
        OPENVINO_TYPE_CASE(evaluate_shape_of, u64, input_shape, output_value);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool constant_fold_shape_of(Node* shape_of_node, Output<Node>& replacement, const Output<Node>& shape_of_input) {
    const auto& partial_shape = shape_of_input.get_partial_shape();
    if (partial_shape.is_static()) {
        const auto& output_type = shape_of_node->get_output_element_type(0);
        const auto& output_shape = shape_of_node->get_output_shape(0);
        auto result_tensor = ov::Tensor{output_type, output_shape};
        if (evaluate_shape_of(result_tensor, shape_of_input.get_shape())) {
            replacement = std::make_shared<v0::Constant>(result_tensor);
            return true;
        }
    }
    return false;
}

bool evaluate_bound(const Node* const node, ov::TensorVector& outputs, const bool is_upper) {
    OPENVINO_ASSERT(outputs.size() == 1);
    const auto& in_shape = node->get_input_partial_shape(0);

    if (in_shape.rank().is_static()) {
        const auto& out_et = outputs[0].get_element_type();
        auto eval_status =
            shape_of::evaluate_shape_of(outputs[0], is_upper ? in_shape.get_max_shape() : in_shape.get_min_shape());

        if (in_shape.size() == 0)
            return eval_status;

        // use node output type as it can be different than output tensor type
        // e.g. when v3::ShapeOf is converted to v0::ShapeOf then the output tensor will have always i64
        // but node output type is transferred from v3 and can be i32 (dimension inf bound is i32 max)
        if (node->get_output_element_type(0) == element::i32) {
            const auto in_shape_rank = in_shape.size();
            constexpr auto max_et_val = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

            const auto get_val = is_upper ? &Interval::get_max_val : &Interval::get_min_val;
            auto limit_val = is_upper ? max_et_val : static_cast<decltype(max_et_val)>(0);

            auto dynamic_mask = std::vector<char>(in_shape_rank);
            std::transform(in_shape.begin(), in_shape.end(), dynamic_mask.begin(), [&](const Dimension& d) {
                return static_cast<char>((d.get_interval().*get_val)() >= max_et_val);
            });

            const auto limit = Tensor(out_et, Shape{}, &limit_val);
            const auto mask = Tensor(element::boolean, Shape{in_shape_rank}, dynamic_mask.data());
            eval_status = v1::Select().evaluate(outputs, {mask, limit, outputs[0]});
        }
        return eval_status;
    } else {
        return false;
    }
}

bool evaluate_symbol(const Node* shape_of_node, TensorSymbolVector& output_symbols) {
    const auto& shape = shape_of_node->get_input_partial_shape(0);
    OPENVINO_ASSERT(shape.rank().is_static());  // sanity check. at this point value propagation was successful

    bool at_least_one_symbol_set = false;
    auto& symbols = output_symbols[0];
    symbols.reserve(shape.size());

    for (const auto& d : shape) {
        const auto symbol = d.get_symbol();
        symbols.emplace_back(symbol);
        at_least_one_symbol_set |= (symbol != nullptr);
    }
    return at_least_one_symbol_set;
}
}  // namespace
}  // namespace shape_of

namespace v3 {
ShapeOf::ShapeOf(const Output<Node>& arg, element::Type output_type) : ShapeOfBase({arg}), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void ShapeOf::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ShapeOf_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");
    set_input_is_relevant_to_value(0, false);
    const auto& input_partial_shape = get_input_partial_shape(0);
    set_output_type(0, m_output_type, PartialShape{input_partial_shape.rank()});
}

bool ShapeOf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ShapeOf_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<Node> ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ShapeOf>(new_args.at(0), m_output_type);
}

bool ShapeOf::evaluate(TensorVector& output_values, const TensorVector& input_values) const {
    OV_OP_SCOPE(v0_ShapeOf_evaluate);
    OPENVINO_ASSERT(input_values.size() == 1);
    OPENVINO_ASSERT(output_values.size() == 1);

    return shape_of::evaluate_shape_of(output_values[0], input_values[0].get_shape());
}

bool ShapeOf::has_evaluate() const {
    OV_OP_SCOPE(v3_ShapeOf_has_evaluate);
    switch (get_output_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool ShapeOf::evaluate_lower(ov::TensorVector& output_values) const {
    return shape_of::evaluate_bound(this, output_values, false);
}

bool ShapeOf::evaluate_upper(ov::TensorVector& output_values) const {
    return shape_of::evaluate_bound(this, output_values, true);
}

bool ShapeOf::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return shape_of::evaluate_symbol(this, output_symbols);
}

bool ShapeOf::can_constant_fold(const OutputVector& input_values) const {
    return !is_const_fold_disabled() && input_values[0].get_partial_shape().is_static();
}

bool ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v3_ShapeOf_constant_fold);
    if (!can_constant_fold(input_values)) {
        return false;
    }
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0]);
}
}  // namespace v3

namespace v0 {
ShapeOf::ShapeOf(const Output<Node>& arg) : ShapeOfBase({arg}) {
    constructor_validate_and_infer_types();
}

void ShapeOf::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ShapeOf_validate_and_infer_types);
    set_input_is_relevant_to_value(0, false);
    set_output_type(0, element::i64, PartialShape{get_input_partial_shape(0).rank()});
}

std::shared_ptr<Node> ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_shape_of = std::make_shared<v0::ShapeOf>(new_args.at(0));
    OPENVINO_ASSERT(new_shape_of.get(),
                    new_shape_of != nullptr,
                    "Cannot clone ",
                    description(),
                    " operation with name ",
                    get_friendly_name());
    return new_shape_of;
}

bool ShapeOf::evaluate(TensorVector& output_values, const TensorVector& input_values) const {
    OV_OP_SCOPE(v0_ShapeOf_evaluate);
    OPENVINO_ASSERT(input_values.size() == 1);
    OPENVINO_ASSERT(output_values.size() == 1);

    return shape_of::evaluate_shape_of(output_values[0], input_values[0].get_shape());
}

bool ShapeOf::has_evaluate() const {
    OV_OP_SCOPE(v0_ShapeOf_has_evaluate);
    switch (get_output_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool ShapeOf::can_constant_fold(const OutputVector& input_values) const {
    return !is_const_fold_disabled() && input_values[0].get_partial_shape().is_static();
}

bool ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v0_ShapeOf_constant_fold);
    if (!can_constant_fold(input_values)) {
        return false;
    }
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0]);
}

bool ShapeOf::evaluate_lower(ov::TensorVector& output_values) const {
    return shape_of::evaluate_bound(this, output_values, false);
}

bool ShapeOf::evaluate_upper(ov::TensorVector& output_values) const {
    return shape_of::evaluate_bound(this, output_values, true);
}

bool ShapeOf::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return shape_of::evaluate_symbol(this, output_symbols);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
