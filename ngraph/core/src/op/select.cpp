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

#include <memory>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/select.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::Select, "Select", 1);

op::v1::Select::Select(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const Output<Node>& arg2,
                       const AutoBroadcastSpec& auto_broadcast)
    : Op({arg0, arg1, arg2})
    , m_auto_broadcast(auto_broadcast)
{
    constructor_validate_and_infer_types();
}

void op::v1::Select::validate_and_infer_types()
{
    // Condition element type check
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() ||
                              get_input_element_type(0) == element::boolean,
                          "Argument 0 must have boolean element type (element type: ",
                          get_input_element_type(0),
                          ").");

    // Then/Else element type check
    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, get_input_element_type(1), get_input_element_type(2)),
        "Argument 1 and 2 element types must match.");

    PartialShape result_shape = get_input_partial_shape(2);
    for (int i = 1; i >= 0; i--)
    {
        if (get_auto_broadcast().m_type == op::AutoBroadcastType::NONE)
        {
            NODE_VALIDATION_CHECK(
                this,
                PartialShape::merge_into(result_shape, get_input_partial_shape(i)),
                "Argument shapes are inconsistent.");
        }
        else if (get_auto_broadcast().m_type == op::AutoBroadcastType::NUMPY ||
                 get_auto_broadcast().m_type == op::AutoBroadcastType::PDPD)
        {
            NODE_VALIDATION_CHECK(this,
                                  PartialShape::broadcast_merge_into(result_shape,
                                                                     get_input_partial_shape(i),
                                                                     get_auto_broadcast()),
                                  "Argument shapes are inconsistent.");
        }
        else
        {
            NODE_VALIDATION_CHECK(this, false, "Unsupported auto broadcast specification");
        }
    }
    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::v1::Select::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Select>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_auto_broadcast);
}

bool op::v1::Select::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

constexpr NodeTypeInfo op::v0::Select::type_info;

op::v0::Select::Select(const Output<Node>& arg0, const Output<Node>& arg1, const Output<Node>& arg2)
    : Op({arg0, arg1, arg2})
{
    constructor_validate_and_infer_types();
}

void op::v0::Select::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() ||
                              get_input_element_type(0) == element::boolean,
                          "Argument 0 must have boolean element type (element type: ",
                          get_input_element_type(0),
                          ").");

    PartialShape result_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          PartialShape::merge_into(result_shape, get_input_partial_shape(1)),
                          "Argument shapes are inconsistent.");
    NODE_VALIDATION_CHECK(this,
                          PartialShape::merge_into(result_shape, get_input_partial_shape(2)),
                          "Argument shapes are inconsistent.");

    element::Type result_et;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, get_input_element_type(1), get_input_element_type(2)),
        "Argument 1 and 2 element types are inconsistent.");

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::v0::Select::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::Select>(new_args.at(0), new_args.at(1), new_args.at(2));
}
