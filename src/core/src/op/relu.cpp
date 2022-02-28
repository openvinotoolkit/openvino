// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/relu.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "openvino/core/evaluate_extension.hpp"

namespace {

const ov::EvaluateExtension::Ptr get_evaluate_extension(const ov::DiscreteTypeInfo& type) {
    for (const auto ext : ov::get_extensions_for_type(type)) {
        if (auto eval_ext = std::dynamic_pointer_cast<ov::EvaluateExtension>(ext)) {
            return eval_ext;
        }
    }
    return nullptr;
}

}  // namespace

BWDCMP_RTTI_DEFINITION(ov::op::v0::Relu);

ov::op::v0::Relu::Relu(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v0::Relu::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v0_Relu_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Relu>(new_args.at(0));
}

namespace relu {
namespace {
template <ov::element::Type_t ET>
inline bool evaluate(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out, const size_t count) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::relu<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_relu(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(arg0->get_shape());
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_relu, i32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_relu, i64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_relu, u32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_relu, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_relu, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_relu, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace relu

bool ov::op::v0::Relu::evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v0_Relu_evaluate);
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(outputs, 1) && ngraph::validate_host_tensor_vector(inputs, 1));
    return relu::evaluate_relu(inputs[0], outputs[0]);
}

bool ov::op::v0::Relu::has_evaluate() const {
    NGRAPH_OP_SCOPE(v0_Relu_has_evaluate);
    auto ext = get_evaluate_extension(get_type_info());
    if (ext)
        return ext->has_evaluate(shared_from_this());
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

bool ov::op::v0::Relu::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v0_Relu_visit_attributes);
    return true;
}
