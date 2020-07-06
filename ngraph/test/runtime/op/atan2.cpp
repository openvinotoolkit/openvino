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

#include "atan2.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Atan2::type_info;

op::v0::Atan2::Atan2(const Output<Node>& y, const Output<Node>& x, const AutoBroadcastSpec& autob)
    : BinaryElementwiseArithmetic(y, x, autob)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Atan2::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Atan2>(new_args.at(0), new_args.at(1), this->get_autob());
}

void op::v0::Atan2::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }
    auto y = input_value(0);
    auto x = input_value(1);
    auto delta_over_r = deltas.at(0) / (x * x + y * y);
    adjoints.add_delta(y, x * delta_over_r);
    adjoints.add_delta(x, -y * delta_over_r);
}

bool op::v0::Atan2::visit_attributes(AttributeVisitor& visitor)
{
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    return true;
}
