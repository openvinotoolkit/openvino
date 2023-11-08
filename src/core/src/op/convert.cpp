// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/reference/convert.hpp"
#include "validation_util.hpp"

namespace ov {
namespace op {
namespace convert {

constexpr bool is_lp_type(const element::Type_t et) {
    return (et == element::i4) || (et == element::u1) || (et == element::u4) || (et == element::nf4);
}

#define CONVERT_ET_LIST boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u4, u8, u16, u32, u64, nf4

struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;
    template <element::Type_t ET, class TO = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        using namespace ov::element;
        return IfTypeOf<CONVERT_ET_LIST>::apply<EvalByOutputType<is_lp_type(ET)>>(
            out.get_element_type(),
            reinterpret_cast<const TO*>(arg.data()),
            out,
            count,
            ET);
    }

private:
    template <bool IS_ARG_ET_LP>
    struct EvalByOutputType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t ET,
                  class T,
                  class T_ET,
                  class U = fundamental_type_for<ET>,
                  typename std::enable_if<is_lp_type(ET) || IS_ARG_ET_LP>::type* = nullptr>
        static result_type visit(const T* arg, Tensor& out, const size_t count, T_ET&& arg_et) {
            reference::detail::lp_convert(arg, reinterpret_cast<U*>(out.data()), count, arg_et, ET);
            return true;
        }

        template <element::Type_t ET,
                  class T,
                  class T_ET,
                  class U = fundamental_type_for<ET>,
                  typename std::enable_if<!is_lp_type(ET) && !IS_ARG_ET_LP>::type* = nullptr>
        static result_type visit(const T* arg, Tensor& out, const size_t count, T_ET&&) {
            reference::convert(arg, out.data<U>(), count);
            return true;
        }
    };
};

namespace {
bool evaluate_bound(const Node* const node, TensorVector& output_values, const Tensor& input_bound) {
    OPENVINO_ASSERT(node, output_values.size() == 1);

    if (input_bound) {
        const auto& in_bound_shape = input_bound.get_shape();
        if (is_vector(in_bound_shape) && (in_bound_shape[0] == 0)) {
            return true;
        }

        const auto is_integral_up_to_16_bits = [](const element::Type& et) -> bool {
            return et.is_integral() && et.bitwidth() <= 16;
        };
        const auto& input_et = input_bound.get_element_type();
        const auto& output_et = output_values[0].get_element_type();

        const auto status = node->evaluate(output_values, {input_bound});
        if (!status || is_integral_up_to_16_bits(input_et) || is_integral_up_to_16_bits(output_et)) {
            return status;
        }

        // constants for dynamic values translation
        const auto input_max = ov::util::make_tensor_of_max_value(input_et);
        const auto output_max = ov::util::make_tensor_of_max_value(output_et);
        if (!input_max || !output_max)
            return false;

        // dynamic values translation
        auto input_dynamic_mask = Tensor(element::boolean, in_bound_shape);
        auto outputs = TensorVector{input_dynamic_mask};

        return v1::Equal().evaluate(outputs, {input_bound, input_max}) &&
               v1::Select().evaluate(output_values, {input_dynamic_mask, output_max, output_values[0]});
    } else {
        return false;
    }
}
}  // namespace
}  // namespace convert
namespace v0 {

Convert::Convert(const Output<Node>& arg, const element::Type& destination_type)
    : Op({arg}),
      m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

void Convert::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Convert_validate_and_infer_types);

    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

bool Convert::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Convert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

std::shared_ptr<Node> Convert::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Convert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Convert>(new_args.at(0), m_destination_type);
}

bool Convert::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Convert_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& in_shape = inputs[0].get_shape();
    outputs[0].set_shape(in_shape);
    using namespace ov::element;
    return IfTypeOf<CONVERT_ET_LIST>::apply<convert::Evaluate>(inputs[0].get_element_type(),
                                                               inputs[0],
                                                               outputs[0],
                                                               shape_size(in_shape));
}

bool Convert::has_evaluate() const {
    OV_OP_SCOPE(v0_Convert_has_evaluate);

    const auto is_valid_type = [](const element::Type& et) -> bool {
        switch (et) {
        case element::boolean:
        case element::bf16:
        case element::f16:
        case element::f32:
        case element::f64:
        case element::i4:
        case element::i8:
        case element::i16:
        case element::i32:
        case element::i64:
        case element::u1:
        case element::u4:
        case element::u8:
        case element::u16:
        case element::u32:
        case element::u64:
        case element::nf4:
            return true;
        default:
            return false;
        };
    };

    return is_valid_type(get_input_element_type(0)) && is_valid_type(get_output_element_type(0));
}

bool Convert::evaluate_lower(TensorVector& output_values) const {
    return convert::evaluate_bound(this, output_values, get_input_tensor(0).get_lower_value());
}

bool Convert::evaluate_upper(TensorVector& output_values) const {
    return convert::evaluate_bound(this, output_values, get_input_tensor(0).get_upper_value());
}

bool Convert::evaluate_label(TensorLabelVector& output_labels) const {
    const auto input_labels = get_input_tensor(0).get_value_label();
    if (input_labels.empty()) {
        return false;
    } else {
        output_labels[0] = input_labels;
        return true;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
