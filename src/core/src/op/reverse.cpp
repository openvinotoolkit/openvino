// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reverse.hpp"

#include <algorithm>
#include <iterator>
#include <sstream>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "openvino/reference/reverse.hpp"
#include "reverse_shape_inference.hpp"

ov::op::v1::Reverse::Reverse(const Output<Node>& data, const Output<Node>& reversed_axes, const std::string& mode)
    : Op({data, reversed_axes}),
      m_mode{mode_from_string(mode)} {
    constructor_validate_and_infer_types();
}

ov::op::v1::Reverse::Reverse(const Output<Node>& data, const Output<Node>& reversed_axes, const Mode mode)
    : Op({data, reversed_axes}),
      m_mode{mode} {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Reverse::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Reverse_visit_attributes);
    visitor.on_attribute("mode", m_mode);
    return true;
}

void ov::op::v1::Reverse::validate_and_infer_types() {
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

    const auto output_shape = shape_infer(this, ov::util::get_node_input_partial_shapes(*this)).front();
    set_output_type(0, get_input_element_type(0), output_shape);
}

std::shared_ptr<ov::Node> ov::op::v1::Reverse::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Reverse_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v1::Reverse>(new_args.at(0), new_args.at(1), m_mode);
}

ov::op::v1::Reverse::Mode ov::op::v1::Reverse::mode_from_string(const std::string& mode) const {
    static const std::map<std::string, Mode> allowed_values = {{"index", Mode::INDEX}, {"mask", Mode::MASK}};

    NODE_VALIDATION_CHECK(this, allowed_values.count(mode) > 0, "Invalid 'mode' value passed in.");

    return allowed_values.at(mode);
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace reverseop {
template <ov::element::Type_t ET>
void get_axes(ov::AxisSet& axes, const ngraph::HostTensorPtr& in) {
    auto axes_indices = in->get_data_ptr<ET>();
    size_t axes_rank = in->get_element_count();
    std::copy(axes_indices, axes_indices + axes_rank, std::inserter(axes, axes.end()));
}
}  // namespace reverseop

#define GET_AXES(a, ...)                                      \
    case element::Type_t::a: {                                \
        OV_OP_SCOPE(OV_PP_CAT3(get_reverse_axes, _, a));      \
        reverseop::get_axes<element::Type_t::a>(__VA_ARGS__); \
    } break;

bool ov::op::v1::Reverse::evaluate_reverse(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    AxisSet axes{};
    if (get_mode() == op::v1::Reverse::Mode::INDEX) {
        switch (inputs[1]->get_element_type()) {
            GET_AXES(i8, axes, inputs[1]);
            GET_AXES(i16, axes, inputs[1]);
            GET_AXES(i32, axes, inputs[1]);
            GET_AXES(i64, axes, inputs[1]);
            GET_AXES(u8, axes, inputs[1]);
            GET_AXES(u16, axes, inputs[1]);
            GET_AXES(u32, axes, inputs[1]);
            GET_AXES(u64, axes, inputs[1]);
        default:
            OPENVINO_ASSERT(false, "Not supported axes type", inputs[1]->get_element_type());
        }
    } else  // Mode::MASK
    {
        auto axes_mask = inputs[1]->get_data_ptr<bool>();
        for (size_t i = 0; i < inputs[1]->get_element_count(); ++i) {
            if (axes_mask[i]) {
                axes.emplace(i);
            }
        }
    }
    ov::reference::reverse(inputs[0]->get_data_ptr<const char>(),
                           outputs[0]->get_data_ptr<char>(),
                           inputs[0]->get_shape(),
                           outputs[0]->get_shape(),
                           axes,
                           inputs[0]->get_element_type().size());
    return true;
}

bool ov::op::v1::Reverse::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Reverse_evaluate);
    return evaluate_reverse(outputs, inputs);
}

bool ov::op::v1::Reverse::has_evaluate() const {
    OV_OP_SCOPE(v1_Reverse_has_evaluate);

    if (get_mode() == op::v1::Reverse::Mode::INDEX) {
        switch (get_input_element_type(1)) {
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64:
            return true;
        default:
            return false;
            ;
        }
    } else {
        return true;
    }
}

std::ostream& ov::operator<<(std::ostream& s, const op::v1::Reverse::Mode& type) {
    return s << as_string(type);
}

namespace ov {
template <>
NGRAPH_API EnumNames<ngraph::op::v1::Reverse::Mode>& EnumNames<ngraph::op::v1::Reverse::Mode>::get() {
    static auto enum_names = EnumNames<ngraph::op::v1::Reverse::Mode>(
        "op::v1::Reverse::Mode",
        {{"index", ngraph::op::v1::Reverse::Mode::INDEX}, {"mask", ngraph::op::v1::Reverse::Mode::MASK}});
    return enum_names;
}
}  // namespace ov
