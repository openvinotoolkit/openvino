// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/non_zero.hpp"

#include <ngraph/validation_util.hpp>
#include <numeric>

#include "itt.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "openvino/reference/non_zero.hpp"

using namespace ngraph;
using namespace std;

op::v3::NonZero::NonZero(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

op::v3::NonZero::NonZero(const Output<Node>& arg, const std::string& output_type)
    : Op({arg}),
      m_output_type(EnumNames<element::Type_t>::as_enum(output_type)) {
    constructor_validate_and_infer_types();
}

op::v3::NonZero::NonZero(const Output<Node>& arg, const element::Type& output_type)
    : Op({arg}),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v3::NonZero::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_NonZero_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v3::NonZero::validate_and_infer_types() {
    OV_OP_SCOPE(v3_NonZero_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");
    // For scalar non-zero value case, onnx test case expects output shape {1, 1}
    const ov::PartialShape& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().compatible(0)) {
        set_output_type(0, m_output_type, ov::PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    } else {
        const Dimension dim =
            std::accumulate(begin(input_shape), end(input_shape), Dimension(0, 1), std::multiplies<Dimension>());
        set_output_type(0, m_output_type, ov::PartialShape{input_shape.rank(), dim});
    }

    set_input_is_relevant_to_shape(0);

    OPENVINO_SUPPRESS_DEPRECATED_START
    if (const auto& input_constant = get_constant_from_source(input_value(0))) {
        // input_value is available to calculate output shape
        const auto& input_data = std::make_shared<HostTensor>(input_constant);
        auto output = std::make_shared<HostTensor>(m_output_type, get_output_partial_shape(0));
        if (!evaluate({output}, {input_data}))
            return;
        set_output_type(0, m_output_type, output->get_partial_shape());

        auto t = Tensor(output->get_element_type(), output->get_shape());
        memcpy(t.data(), output->get_data_ptr(), t.get_byte_size());

        get_output_tensor(0).set_lower_value(t);
        get_output_tensor(0).set_upper_value(t);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

shared_ptr<Node> op::v3::NonZero::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_NonZero_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v3::NonZero>(new_args.at(0), m_output_type);
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace nonzero {
namespace {
template <element::Type_t INPUT_ET, element::Type_t OUT_ET>
bool evaluate_nonzero_execute(const HostTensorPtr& input, const HostTensorPtr& output) {
    using IN_T = typename element_type_traits<INPUT_ET>::value_type;
    using OUT_T = typename element_type_traits<OUT_ET>::value_type;

    ov::Shape input_shape = input->get_shape();
    size_t input_rank = input_shape.size();

    size_t non_zero_count = ov::reference::non_zero_get_count<IN_T>(input->get_data_ptr<INPUT_ET>(), input_shape);

    ov::Shape out_shape;
    if (input_rank == 0 && non_zero_count > 0) {
        out_shape = ov::Shape{1, 1};
    } else {
        out_shape = ov::Shape{input_rank, non_zero_count};
    }

    output->set_shape(out_shape);
    ov::reference::non_zero<IN_T, OUT_T>(input->get_data_ptr<INPUT_ET>(), output->get_data_ptr<OUT_ET>(), input_shape);

    return true;
}

#define TYPE_OUT_CASE(a, ...)                                                     \
    case element::Type_t::a: {                                                    \
        OV_OP_SCOPE(OV_PP_CAT3(evaluate_nonzero_out, _, a));                      \
        rc = evaluate_nonzero_execute<INPUT_ET, element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t INPUT_ET>
bool evaluate(const HostTensorPtr& input, const HostTensorPtr& output) {
    bool rc = true;
    switch (output->get_element_type()) {
        TYPE_OUT_CASE(i64, input, output);
        TYPE_OUT_CASE(i32, input, output);
    default:
        rc = false;
        break;
    }

    return rc;
}
#undef TYPE_OUT_CASE
bool evaluate_nonzero(const HostTensorPtr& input, const HostTensorPtr& output) {
    bool rc = true;

    switch (input->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_nonzero, boolean, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, i8, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, i16, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, i32, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, i64, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, u8, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, u16, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, u32, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, u64, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, bf16, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, f16, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, f32, input, output);
        OPENVINO_TYPE_CASE(evaluate_nonzero, f64, input, output);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace nonzero

bool op::v3::NonZero::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_NonZero_evaluate);
    return nonzero::evaluate_nonzero(inputs[0], outputs[0]);
}

bool op::v3::NonZero::has_evaluate() const {
    OV_OP_SCOPE(v3_NonZero_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
        return true;
    default:
        break;
    }
    return false;
}
