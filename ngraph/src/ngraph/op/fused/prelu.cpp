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
#include "ngraph/op/fused/prelu.hpp"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PRelu::type_info;

op::PRelu::PRelu(const Output<Node>& data, const Output<Node>& slope)
    : FusedOp({data, slope})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::PRelu::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

NodeVector op::PRelu::decompose_op() const
{
    auto data = input_value(0);
    auto data_shape = data.get_shape();
    auto slope = input_value(1);
    slope = std::make_shared<op::Convert>(slope, data.get_element_type());
    auto slope_shape = slope.get_shape();

    if ((slope_shape.size() == 1) && (slope_shape.at(0) != 1))
    {
        auto it = std::find(std::begin(data_shape), std::end(data_shape), slope_shape.at(0));
        auto index = std::distance(std::begin(data_shape), it);
        slope = builder::make_broadcast_node(slope, data.get_shape(), index);
    }
    else if (data_shape != slope_shape)
    {
        slope = builder::numpy_broadcast(slope, data.get_shape());
    }

    // x <  0 => f(x) = x * slope
    // x >= 0 => f(x) = x

    std::shared_ptr<ngraph::Node> zero_node = std::make_shared<ngraph::op::Constant>(
        data.get_element_type(), ngraph::Shape{}, std::vector<double>{0});
    zero_node = builder::make_broadcast_node(zero_node, data.get_shape());

    std::shared_ptr<ngraph::Node> negative_map = std::make_shared<ngraph::op::Convert>(
        std::make_shared<ngraph::op::Less>(data, zero_node), data.get_element_type());

    std::shared_ptr<ngraph::Node> positive_map = std::make_shared<ngraph::op::Convert>(
        std::make_shared<ngraph::op::Greater>(data, zero_node), data.get_element_type());

    slope = negative_map * slope + positive_map;

    return {data * slope};
}

shared_ptr<Node> op::PRelu::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<PRelu>(new_args.at(0), new_args.at(1));
}
