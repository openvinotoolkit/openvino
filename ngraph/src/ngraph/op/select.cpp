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

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/select.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::Select::type_info;

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

void op::v1::Select::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_auto_broadcast().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto p = input_value(0);
    auto x = input_value(1);
    auto y = input_value(2);

    auto p_as_x_type = make_shared<op::Convert>(p, x.get_element_type());
    auto not_p_as_y_type = make_shared<op::Convert>(make_shared<op::Not>(p), y.get_element_type());

    adjoints.add_delta(x, delta * p_as_x_type);
    adjoints.add_delta(y, delta * not_p_as_y_type);
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

void op::Select::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto p = input_value(0);
    auto x = input_value(1);
    auto y = input_value(2);

    auto p_as_x_type = make_shared<op::Convert>(p, x.get_element_type());
    auto not_p_as_y_type = make_shared<op::Convert>(make_shared<op::Not>(p), y.get_element_type());

    adjoints.add_delta(x, delta * p_as_x_type);
    adjoints.add_delta(y, delta * not_p_as_y_type);
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& arg2,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::select(arg0->get_data_ptr<element::Type_t::boolean>(),
                                   arg1->get_data_ptr<ET>(),
                                   arg2->get_data_ptr<ET>(),
                                   out->get_data_ptr<ET>(),
                                   arg0->get_shape(),
                                   arg1->get_shape(),
                                   arg2->get_shape(),
                                   broadcast_spec);
        return true;
    }

    bool evaluate_select(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& arg2,
                         const HostTensorPtr& out,
                         const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_shape(arg0->get_shape());
        out->set_element_type(arg1->get_element_type()); // assume that arg1 and arg2 are already
                                                         // validation and consistent
        switch (arg1->get_element_type())
        {
            TYPE_CASE(i8)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(i16)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(i32)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(i64)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(u8)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(u16)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(u32)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(u64)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(bf16)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(f16)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(f32)(arg0, arg1, arg2, out, broadcast_spec);
            break;
            TYPE_CASE(f64)(arg0, arg1, arg2, out, broadcast_spec);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v1::Select::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_select(inputs[0], inputs[1], inputs[2], outputs[0], get_autob());
}
