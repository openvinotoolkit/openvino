//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/squared_difference.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/fused_op.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::SquaredDifference::type_info;

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
    visitor.on_attribute("auto_broadcast", m_autobroadcast);
    return true;
}

OutputVector op::SquaredDifference::decompose_op() const
{
    const auto x1 = input_value(0);
    const auto x2 = input_value(1);

    const auto difference = make_shared<op::Subtract>(x1, x2, m_autobroadcast);

    return {difference * difference};
}

shared_ptr<Node> op::SquaredDifference::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);

    return make_shared<SquaredDifference>(new_args.at(0), new_args.at(1), get_autob());
}
