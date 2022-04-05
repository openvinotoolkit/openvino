// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softsign.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"

#include <cmath>
#include <cstddef>

namespace ov {
namespace intel_gna {
namespace op {

template <typename T>
void softsign(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = arg[i] / (1 + std::abs(arg[i]));
    }
}

SoftSign::SoftSign(const ngraph::Output<ngraph::Node>& arg) : ov::op::util::UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> SoftSign::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SoftSign>(new_args.at(0));
}

template <ngraph::element::Type_t ET>
inline bool evaluate(const ov::Tensor& arg, ov::Tensor& out, const size_t count) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    softsign<T>(arg.data<T>(), out.data<T>(), count);
    return true;
}

namespace {
bool evaluate_softsign(const ov::Tensor& arg, ov::Tensor& out) {
    bool rc = true;
    size_t count = shape_size(arg.get_shape());

    switch (arg.get_element_type()) {
    case ov::element::Type_t::f16:
        rc = evaluate<ov::element::Type_t::f16>(arg, out, count);
        break;
    case ov::element::Type_t::f32:
        rc = evaluate<ov::element::Type_t::f32>(arg, out, count);
        break;
    default:
        rc = false;
        break;
    }
    return rc;
}
} // namespace

bool SoftSign::evaluate(ov::TensorVector& outputs,
                        const ov::TensorVector& inputs,
                        const ov::EvaluationContext& evaluation_context) const {
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

} // namespace op
} // namespace intel_gna
} // namespace ov
