// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softsign.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"

#include <cmath>
#include <cstddef>

NGRAPH_RTTI_DEFINITION(GNAPluginNS::SoftSign, "SoftSign", 0);

namespace GNAPluginNS {

template <typename T>
void softsign(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = 1 / (1 + std::abs(arg[i]));
    }
}

SoftSign::SoftSign(const ngraph::Output<ngraph::Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

bool SoftSign::visit_attributes(ngraph::AttributeVisitor& visitor) {
    return true;
}

void SoftSign::validate_and_infer_types() {
    const ngraph::element::Type& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be float. Got: ",
                          input_et);

    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ngraph::Node> SoftSign::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SoftSign>(new_args.at(0));
}

template <ngraph::element::Type_t ET>
inline bool evaluate(const ngraph::HostTensorPtr& arg, const ngraph::HostTensorPtr& out, const size_t count) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    softsign<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

#ifndef NGRAPH_TYPE_CASE
/* NGRAPH_TYPE_CASE is basicaly defined in ngraph/core/src/itt.h and also contains different
   macro for tracing
   TODO: is it good to use that header since it's not placed in ngraph/core/include?
*/
#define NGRAPH_TYPE_CASE(region, a, ...)                        \
    case ov::element::Type_t::a: {                              \
        {                                                       \
            rc = evaluate<ov::element::Type_t::a>(__VA_ARGS__); \
        }                                                       \
    } break
#endif // NGRAPH_TYPE_CASE

bool evaluate_softsign(const ngraph::HostTensorPtr& arg, const ngraph::HostTensorPtr& out) {
    bool rc = true;
    out->set_unary(arg);
    size_t count = shape_size(arg->get_shape());

    switch (arg->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_softsign, f16, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softsign, f32, arg, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool SoftSign::evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const {
    return evaluate_softsign(inputs[0], outputs[0]);
}

bool SoftSign::has_evaluate() const {
    switch (get_input_element_type(0)) {
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

} // namespace GNAPluginNS
