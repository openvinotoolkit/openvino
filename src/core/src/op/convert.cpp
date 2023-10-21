// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/convert.hpp"

#include <memory>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/select.hpp"
#include "openvino/reference/convert.hpp"

using namespace std;
using namespace ngraph;

op::Convert::Convert(const Output<Node>& arg, const element::Type& destination_type)
    : Op({arg}),
      m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

void op::Convert::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Convert_validate_and_infer_types);

    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

bool op::Convert::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Convert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

shared_ptr<Node> op::Convert::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Convert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Convert>(new_args.at(0), m_destination_type);
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace convert {
namespace {
template <element::Type_t INPUT_ET, element::Type_t OUTPUT_ET>
bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out) {
    out->set_shape(arg->get_shape());
    size_t element_count = shape_size(out->get_shape());

    if ((INPUT_ET != arg->get_element_type()) || OUTPUT_ET != out->get_element_type()) {
        return false;
    }
    if (((INPUT_ET == element::u1) || (OUTPUT_ET == element::u1)) ||
        ((INPUT_ET == element::u4) || (OUTPUT_ET == element::u4)) ||
        ((INPUT_ET == element::i4) || (OUTPUT_ET == element::i4)) ||
        ((INPUT_ET == element::nf4) || (OUTPUT_ET == element::nf4))) {
        ov::reference::detail::lp_convert(arg->get_data_ptr<INPUT_ET>(),
                                          out->get_data_ptr<OUTPUT_ET>(),
                                          element_count,
                                          INPUT_ET,
                                          OUTPUT_ET);
    } else {
        ov::reference::convert(arg->get_data_ptr<INPUT_ET>(), out->get_data_ptr<OUTPUT_ET>(), element_count);
    }
    return true;
}

#define TYPE_OUT_CASE(a, ...)                                     \
    case element::Type_t::a: {                                    \
        OV_OP_SCOPE(OV_PP_CAT3(evaluate_covert_out, _, a));       \
        rc = evaluate<INPUT_ET, element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t INPUT_ET>
bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out) {
    bool rc = true;

    switch (out->get_element_type()) {
        TYPE_OUT_CASE(i4, arg, out);
        TYPE_OUT_CASE(i8, arg, out);
        TYPE_OUT_CASE(i16, arg, out);
        TYPE_OUT_CASE(i32, arg, out);
        TYPE_OUT_CASE(i64, arg, out);
        TYPE_OUT_CASE(u1, arg, out);
        TYPE_OUT_CASE(u4, arg, out);
        TYPE_OUT_CASE(u8, arg, out);
        TYPE_OUT_CASE(u16, arg, out);
        TYPE_OUT_CASE(u32, arg, out);
        TYPE_OUT_CASE(u64, arg, out);
        TYPE_OUT_CASE(bf16, arg, out);
        TYPE_OUT_CASE(f16, arg, out);
        TYPE_OUT_CASE(f32, arg, out);
        TYPE_OUT_CASE(f64, arg, out);
        TYPE_OUT_CASE(boolean, arg, out);
        TYPE_OUT_CASE(nf4, arg, out);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_convert(const HostTensorPtr& arg, const HostTensorPtr& out) {
    bool rc = true;
    switch (arg->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_convert, u1, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, u4, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, u8, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, u16, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, u32, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, u64, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, i4, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, i8, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, i16, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, i32, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, i64, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, bf16, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, f16, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, f32, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, f64, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, boolean, arg, out);
        OPENVINO_TYPE_CASE(evaluate_convert, nf4, arg, out);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_bound(const Node* node, ov::TensorVector& output_values, bool is_upper) {
    OPENVINO_ASSERT(node, output_values.size() == 1);
    const auto& input = node->input_value(0);
    if (const auto& value = is_upper ? input.get_tensor().get_upper_value() : input.get_tensor().get_lower_value()) {
        if (is_vector(value.get_shape()) && (value.get_shape().front() == 0)) {
            return true;
        }
        bool status = node->evaluate(output_values, {value});

        if (!status)
            return status;

        const auto& input_element_type = input.get_element_type();
        const auto& output_element_type = output_values[0].get_element_type();
        if ((input_element_type.is_integral() && input_element_type.bitwidth() <= 16) ||
            (output_element_type.is_integral() && output_element_type.bitwidth() <= 16)) {
            return status;
        }

        // constants for dynamic values translation
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto input_maximum_value = get_constant_max_of_type(input_element_type);
        auto output_maximum_value = get_constant_max_of_type(output_values[0].get_element_type());
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (input_maximum_value == nullptr || output_maximum_value == nullptr)
            return false;

        auto input_max = ov::Tensor(input_maximum_value->get_element_type(), input_maximum_value->get_shape());
        memcpy(input_max.data(), input_maximum_value->get_data_ptr(), input_max.get_byte_size());

        auto output_max = ov::Tensor(output_maximum_value->get_element_type(), output_maximum_value->get_shape());
        memcpy(output_max.data(), output_maximum_value->get_data_ptr(), output_max.get_byte_size());

        // dynamic values translation
        auto input_dynamic_mask = ov::Tensor(element::boolean, input.get_shape());
        auto outputs = ov::TensorVector{input_dynamic_mask};

        status = op::v1::Equal().evaluate(outputs, {value, input_max});
        if (!status)
            return status;

        status = op::v1::Select().evaluate(output_values, {input_dynamic_mask, output_max, output_values[0]});
        return status;
    } else
        return false;
}
}  // namespace
}  // namespace convert
bool op::v0::Convert::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v0_Convert_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(input_values, 1));
    OPENVINO_ASSERT(validate_host_tensor_vector(output_values, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return convert::evaluate_convert(input_values[0], output_values[0]);
}

bool op::v0::Convert::has_evaluate() const {
    OV_OP_SCOPE(v0_Convert_has_evaluate);

    switch (get_input_element_type(0)) {
    case ngraph::element::u1:
    case ngraph::element::u4:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::i4:
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::boolean:
        break;
    default:
        return false;
    }
    switch (get_output_element_type(0)) {
    case ngraph::element::i4:
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u1:
    case ngraph::element::u4:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::boolean:
        break;
    default:
        return false;
    }
    return true;
}

bool op::v0::Convert::evaluate_lower(ov::TensorVector& output_values) const {
    return convert::evaluate_bound(this, output_values, false);
}

bool op::v0::Convert::evaluate_upper(ov::TensorVector& output_values) const {
    return convert::evaluate_bound(this, output_values, true);
}

bool op::v0::Convert::evaluate_label(TensorLabelVector& output_labels) const {
    const auto input_labels = get_input_tensor(0).get_value_label();
    if (input_labels.empty())
        return false;
    output_labels[0] = input_labels;
    return true;
}
