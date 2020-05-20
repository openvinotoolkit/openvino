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
#include "ngraph/op/fused/selu.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Selu::type_info;

op::v0::Selu::Selu(const Output<Node>& data, const Output<Node>& alpha, const Output<Node>& lambda)
    : FusedOp({data, alpha, lambda})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Selu::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

NodeVector op::v0::Selu::decompose_op() const
{
    const auto data = input_value(0);
    const auto alpha = input_value(1);
    const auto lambda = input_value(2);
    const auto zero_node = op::Constant::create(data.get_element_type(), Shape{1}, {0});

    // lambda * ((max(data, 0) + (alpha * exp(min(data, 0)) - alpha))
    return {std::make_shared<op::v1::Multiply>(
        lambda,
        std::make_shared<op::v1::Add>(
            std::make_shared<op::v1::Maximum>(data, zero_node),
            std::make_shared<op::v1::Subtract>(
                std::make_shared<op::v1::Multiply>(
                    alpha,
                    std::make_shared<op::Exp>(std::make_shared<op::v1::Minimum>(data, zero_node))),
                alpha)))};
}

shared_ptr<Node> op::v0::Selu::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::Selu>(new_args.at(0), new_args.at(1), new_args.at(2));
}
