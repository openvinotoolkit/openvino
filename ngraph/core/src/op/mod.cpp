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
#include "ngraph/op/mod.hpp"
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

op::v1::Mod::Mod(const Output<Node>& A,
                 const Output<Node>& B,
                 const AutoBroadcastSpec& auto_broadcast)
    : FusedOp({A, B})
    , m_auto_broadcast(auto_broadcast)
{
}

bool ngraph::op::v1::Mod::visit_attributes(AttributeVisitor& visitor)
{
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
    auto division =
        make_shared<op::Convert>(make_shared<op::v1::Divide>(dividend, divisor, m_auto_broadcast),
                                 ngraph::element::Type_t::i64);
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
    return make_shared<Mod>(new_args.at(0), new_args.at(1), m_auto_broadcast);
}
