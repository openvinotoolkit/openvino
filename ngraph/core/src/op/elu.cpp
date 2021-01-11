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

#include "ngraph/op/elu.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Elu::type_info;

op::Elu::Elu(const Output<Node>& data, const double alpha)
    : Op({data})
    , m_alpha{alpha}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Elu::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Elu_visit_attributes);
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

void op::v0::Elu::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_Elu_validate_and_infer_types);
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::Elu::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Elu_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Elu>(new_args.at(0), m_alpha);
}
