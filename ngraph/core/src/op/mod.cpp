// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/mod.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::v1::Mod::type_info;

op::v1::Mod::Mod()
    : FusedOp()
    , m_auto_broadcast()
{
}

op::v1::Mod::Mod(const Output<Node>& A,
                 const Output<Node>& B,
                 const AutoBroadcastSpec& auto_broadcast)
    : FusedOp({A, B})
    , m_auto_broadcast(auto_broadcast)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Mod::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Mod_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

OutputVector op::v1::Mod::decompose_op() const
{
    const auto dividend = make_shared<op::Abs>(input_value(0));
    const auto dividend_sign = make_shared<op::Sign>(input_value(0));
    const auto dividend_et = dividend->get_element_type();
    const auto divisor = make_shared<op::Abs>(input_value(1));

    // truncated(a / b)
    auto division = make_shared<op::Convert>(
        make_shared<op::v1::Divide>(dividend, divisor, m_auto_broadcast), ngraph::element::i64);
    division = make_shared<op::Convert>(division, dividend_et);
    // truncated(a / b) * b
    const auto multiplication = make_shared<op::v1::Multiply>(division, divisor, m_auto_broadcast);
    // a mod b = a - truncated(a / b) * b
    const auto mod = make_shared<op::v1::Subtract>(dividend, multiplication, m_auto_broadcast);

    // apply sign of dividend
    return {make_shared<op::v1::Multiply>(dividend_sign, mod, m_auto_broadcast)};
}

shared_ptr<Node> op::v1::Mod::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Mod_clone_with_new_inputs);
    return make_shared<Mod>(new_args.at(0), new_args.at(1), m_auto_broadcast);
}
