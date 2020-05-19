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

#include "ngraph/op/argmax.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ArgMax::type_info;

op::ArgMax::ArgMax(const Output<Node>& arg, size_t axis, const element::Type& index_element_type)
    : op::util::IndexReduction(arg, axis, index_element_type)
{
    constructor_validate_and_infer_types();
}

bool op::ArgMax::visit_attributes(AttributeVisitor& visitor)
{
    IndexReduction::visit_attributes(visitor);
    return true;
}

shared_ptr<Node> op::ArgMax::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ArgMax>(new_args.at(0), m_axis, this->get_element_type());
}

std::shared_ptr<Node> op::ArgMax::get_default_value() const
{
    // Choice of value here is arbitrary, because validation should be rejecting cases where the
    // axis of reduction has size zero.
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
