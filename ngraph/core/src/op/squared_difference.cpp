// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/squared_difference.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/fused_op.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::SquaredDifference::type_info;

op::SquaredDifference::SquaredDifference()
    : FusedOp()
    , m_autobroadcast(AutoBroadcastType::NUMPY)
{
}

op::SquaredDifference::SquaredDifference(const Output<Node>& x1,
                                         const Output<Node>& x2,
                                         const AutoBroadcastSpec& auto_broadcast)
    : FusedOp({x1, x2})
    , m_autobroadcast(auto_broadcast)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::SquaredDifference::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_SquaredDifference_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_autobroadcast);
    return true;
}

OutputVector op::SquaredDifference::decompose_op() const
{
    const auto x1 = input_value(0);
    const auto x2 = input_value(1);

    const auto difference = make_shared<op::v1::Subtract>(x1, x2, m_autobroadcast);

    return {make_shared<op::v1::Multiply>(difference, difference)};
}

shared_ptr<Node> op::SquaredDifference::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_SquaredDifference_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return make_shared<SquaredDifference>(new_args.at(0), new_args.at(1), get_autob());
}
