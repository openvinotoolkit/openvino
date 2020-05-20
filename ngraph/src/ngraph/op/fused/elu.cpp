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
#include "ngraph/op/fused/elu.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Elu::type_info;

op::Elu::Elu(const Output<Node>& data, const double alpha)
    : FusedOp({data})
    , m_alpha{alpha}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Elu::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

NodeVector op::Elu::decompose_op() const
{
    auto data = input_value(0);
    shared_ptr<Node> alpha_node =
        make_shared<op::Constant>(data.get_element_type(), Shape{}, vector<double>{m_alpha});

    alpha_node = builder::numpy_broadcast(alpha_node, data.get_shape());

    shared_ptr<ngraph::Node> zero_node =
        builder::make_constant(data.get_element_type(), data.get_shape(), 0);

    return {make_shared<ngraph::op::Maximum>(data, zero_node) +
            alpha_node *
                make_shared<ngraph::op::Exp>(make_shared<ngraph::op::Minimum>(data, zero_node)) -
            alpha_node};
}

shared_ptr<Node> op::Elu::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Elu>(new_args.at(0), m_alpha);
}
