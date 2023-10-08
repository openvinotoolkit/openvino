// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/mish.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/reference/mish.hpp"

using namespace std;
using namespace ngraph;

op::v4::Mish::Mish(const Output<Node>& arg) : util::UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool op::v4::Mish::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_Mish_visit_attributes);
    return true;
}

void op::v4::Mish::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Mish_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Only accepts one argument. Got: ", get_input_size());

    element::Type data_batch_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_batch_et.is_real(),
                          "Element must be of floating point type, Got: ",
                          data_batch_et);

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v4::Mish::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Mish_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Mish>(new_args.at(0));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace mish {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    ov::reference::mish<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_mish(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(arg0->get_shape());
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_mish, f16, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_mish, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace mish

bool op::v4::Mish::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v4_Mish_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return mish::evaluate_mish(inputs[0], outputs[0]);
}

bool op::v4::Mish::has_evaluate() const {
    OV_OP_SCOPE(v4_Mish_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
