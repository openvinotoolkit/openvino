// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/round.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/eval_copy.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/runtime/reference/round.hpp"

using namespace std;
using namespace ngraph;

namespace roundop {
namespace {
// function used by TYPE_CASE
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0,
                     const HostTensorPtr& out,
                     const size_t count,
                     const op::v5::Round::RoundMode mode) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::round<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count, mode);
    return true;
}

// function used by COPY_TENSOR
template <element::Type_t ET>
inline bool copy_tensor(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    runtime::reference::copy(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_round(const HostTensorPtr& arg0,
                    const HostTensorPtr& out,
                    const size_t count,
                    const op::v5::Round::RoundMode mode) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_COPY_TENSOR(evaluate_round, boolean, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, i8, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, i16, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, i32, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, i64, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, u8, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, u16, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, u32, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_round, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_round, f16, arg0, out, count, mode);
        NGRAPH_TYPE_CASE(evaluate_round, f32, arg0, out, count, mode);
        NGRAPH_TYPE_CASE(evaluate_round, bf16, arg0, out, count, mode);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace roundop

BWDCMP_RTTI_DEFINITION(op::v5::Round);

op::v5::Round::Round(const Output<Node>& arg, RoundMode mode) : util::UnaryElementwiseArithmetic(arg), m_mode(mode) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v5::Round::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_Round_visit_attributes);
    visitor.on_attribute("mode", m_mode);
    return true;
}

void op::v5::Round::validate_and_infer_types() {
    OV_OP_SCOPE(v5_Round_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Only accepts one argument. Got: ", get_input_size());
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v5::Round::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_Round_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v5::Round>(new_args.at(0), m_mode);
}

bool op::v5::Round::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v5_Round_evaluate);
    return roundop::evaluate_round(inputs[0], outputs[0], shape_size(get_output_shape(0)), get_mode());
}

bool op::v5::Round::has_evaluate() const {
    OV_OP_SCOPE(v5_Round_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::boolean:
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::bf16:
        return true;
    default:
        break;
    }
    return false;
}

std::ostream& ov::operator<<(std::ostream& s, const op::v5::Round::RoundMode& type) {
    return s << as_string(type);
}

namespace ov {
template <>
NGRAPH_API EnumNames<ngraph::op::v5::Round::RoundMode>& EnumNames<ngraph::op::v5::Round::RoundMode>::get() {
    static auto enum_names = EnumNames<ngraph::op::v5::Round::RoundMode>(
        "op::v5::Round::RoundMode",
        {{"half_to_even", ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN},
         {"half_away_from_zero", ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO}});
    return enum_names;
}

BWDCMP_RTTI_DEFINITION(AttributeAdapter<ov::op::v5::Round::RoundMode>);
}  // namespace ov
