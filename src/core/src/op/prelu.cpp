// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/prelu.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "openvino/reference/prelu.hpp"

using namespace std;

ov::op::v0::PRelu::PRelu() : Op() {}

ov::op::v0::PRelu::PRelu(const Output<Node>& data, const Output<Node>& slope) : Op({data, slope}) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::PRelu::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_PRelu_visit_attributes);
    return true;
}

void ngraph::op::v0::PRelu::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<ov::Node> ov::op::v0::PRelu::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PRelu_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return make_shared<PRelu>(new_args.at(0), new_args.at(1));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace prelu {
namespace {
template <ov::element::Type_t ET>
bool evaluate(const ngraph::HostTensorPtr& arg, const ngraph::HostTensorPtr& slope, const ngraph::HostTensorPtr& out) {
    ov::reference::prelu(arg->get_data_ptr<ET>(),
                         slope->get_data_ptr<ET>(),
                         out->get_data_ptr<ET>(),
                         arg->get_shape(),
                         slope->get_shape());
    return true;
}

bool evaluate_prelu(const ngraph::HostTensorPtr& arg,
                    const ngraph::HostTensorPtr& slope,
                    const ngraph::HostTensorPtr& out) {
    bool rc = true;
    switch (arg->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_prelu, i8, arg, slope, out);
        OPENVINO_TYPE_CASE(evaluate_prelu, bf16, arg, slope, out);
        OPENVINO_TYPE_CASE(evaluate_prelu, f16, arg, slope, out);
        OPENVINO_TYPE_CASE(evaluate_prelu, f32, arg, slope, out);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace prelu

bool ov::op::v0::PRelu::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_PRelu_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1) && ngraph::validate_host_tensor_vector(inputs, 2));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return prelu::evaluate_prelu(inputs[0], inputs[1], outputs[0]);
}

bool ov::op::v0::PRelu::has_evaluate() const {
    OV_OP_SCOPE(v0_PRelu_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
