// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_update.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/reference/scatter_update.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {
ScatterUpdate::ScatterUpdate(const Output<Node>& data,
                             const Output<Node>& indices,
                             const Output<Node>& updates,
                             const Output<Node>& axis)
    : util::ScatterBase(data, indices, updates, axis) {}

std::shared_ptr<Node> ScatterUpdate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ScatterUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ScatterUpdate>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ScatterUpdate::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate);
    OPENVINO_ASSERT(inputs.size() == 4);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    const auto& axis = inputs[3];
    auto& output = outputs[0];

    OPENVINO_ASSERT(axis.get_element_type().is_integral_number(), "axis element type is not integral data type");

    switch (indices.get_element_type()) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        break;
    default:
        return false;
    }

    const auto& data_shape = data.get_shape();
    output.set_shape(data_shape);

    auto axis_val = get_tensor_data_as<int64_t>(axis)[0];
    axis_val = ov::util::try_normalize_axis(axis_val, data_shape.size(), *this);

    const auto indices_casted_vector = get_tensor_data_as<int64_t>(indices);

    reference::scatter_update(static_cast<const char*>(data.data()),
                              indices_casted_vector.data(),
                              static_cast<const char*>(updates.data()),
                              axis_val,
                              static_cast<char*>(output.data()),
                              data.get_element_type().size(),
                              data.get_shape(),
                              indices.get_shape(),
                              updates.get_shape());
    return true;
}

bool ScatterUpdate::evaluate_lower(TensorVector& outputs) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && get_input_tensor(3).has_and_set_bound() &&
           default_lower_bound_evaluator(this, outputs);
}

bool ScatterUpdate::evaluate_upper(TensorVector& outputs) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && get_input_tensor(3).has_and_set_bound() &&
           default_upper_bound_evaluator(this, outputs);
}

bool ScatterUpdate::has_evaluate() const {
    OV_OP_SCOPE(v3_ScatterUpdate_has_evaluate);

    switch (get_input_element_type(1)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool ScatterUpdate::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate_symbol);
    return default_symbol_evaluator(this, {0, 2}, output_symbols);
}
}  // namespace v3
}  // namespace op
}  // namespace ov
