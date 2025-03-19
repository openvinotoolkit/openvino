// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"

#include <sstream>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/reverse.hpp"
#include "reverse_shape_inference.hpp"

namespace ov {
namespace op {
namespace v1 {
namespace {
bool validate_axes_indices_et(const element::Type& et) {
    switch (et) {
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
}  // namespace

Reverse::Reverse(const Output<Node>& data, const Output<Node>& reversed_axes, const std::string& mode)
    : Op({data, reversed_axes}),
      m_mode{mode_from_string(mode)} {
    constructor_validate_and_infer_types();
}

Reverse::Reverse(const Output<Node>& data, const Output<Node>& reversed_axes, const Mode mode)
    : Op({data, reversed_axes}),
      m_mode{mode} {
    constructor_validate_and_infer_types();
}

bool Reverse::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Reverse_visit_attributes);
    visitor.on_attribute("mode", m_mode);
    return true;
}

void Reverse::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Reverse_validate_and_infer_types);
    if (m_mode == Mode::MASK) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(1) == element::boolean,
                              "In 'mask' mode the second input must contain boolean values.");
    } else {
        // Index mode
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(1).is_integral_number(),
                              "In 'index' mode the second input must contain integer values.");
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<ov::Node> Reverse::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Reverse_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Reverse>(new_args.at(0), new_args.at(1), m_mode);
}

Reverse::Mode Reverse::mode_from_string(const std::string& mode) const {
    static const std::map<std::string, Mode> allowed_values = {{"index", Mode::INDEX}, {"mask", Mode::MASK}};

    NODE_VALIDATION_CHECK(this, allowed_values.count(mode) > 0, "Invalid 'mode' value passed in.");

    return allowed_values.at(mode);
}

bool Reverse::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Reverse_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto& data = inputs[0];
    const auto& axes = inputs[1];
    const auto& data_shape = data.get_shape();

    AxisSet reversed_axes{};
    if (get_mode() == Reverse::Mode::MASK) {
        auto axes_mask = axes.data<const fundamental_type_for<element::boolean>>();
        for (size_t i = 0; i < axes.get_size(); ++i, ++axes_mask) {
            if (*axes_mask) {
                reversed_axes.emplace(i);
            }
        }
    } else if (validate_axes_indices_et(axes.get_element_type())) {
        reversed_axes = ov::util::try_get_normalized_axis_set(axes, data_shape.size(), *this);
    } else {
        return false;
    }

    auto& output = outputs[0];
    output.set_shape(data_shape);
    reference::reverse(static_cast<const char*>(data.data()),
                       static_cast<char*>(output.data()),
                       data_shape,
                       output.get_shape(),
                       reversed_axes,
                       data.get_element_type().size());
    return true;
}

bool Reverse::has_evaluate() const {
    OV_OP_SCOPE(v1_Reverse_has_evaluate);
    return (m_mode == Reverse::Mode::MASK) || validate_axes_indices_et(get_input_element_type(1));
}
}  // namespace v1
}  // namespace op

std::ostream& operator<<(std::ostream& s, const op::v1::Reverse::Mode& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v1::Reverse::Mode>& EnumNames<op::v1::Reverse::Mode>::get() {
    static auto enum_names = EnumNames<op::v1::Reverse::Mode>(
        "op::v1::Reverse::Mode",
        {{"index", op::v1::Reverse::Mode::INDEX}, {"mask", op::v1::Reverse::Mode::MASK}});
    return enum_names;
}

AttributeAdapter<op::v1::Reverse::Mode>::~AttributeAdapter() = default;
}  // namespace ov
