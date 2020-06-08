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

#include "ngraph/op/subtract.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/subtract.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------- v0 ------------------------------------------

constexpr NodeTypeInfo op::v0::Subtract::type_info;

op::v0::Subtract::Subtract(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Subtract::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Subtract>(new_args.at(0), new_args.at(1), this->get_autob());
}

void op::Subtract::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);

    adjoints.add_delta(x, delta);
    adjoints.add_delta(y, -delta);
}

shared_ptr<ngraph::Node> ngraph::operator-(const Output<Node> arg0, const Output<Node> arg1)
{
    return make_shared<op::v0::Subtract>(arg0, arg1);
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::subtract(arg0->get_data_ptr<ET>(),
                                     arg1->get_data_ptr<ET>(),
                                     out->get_data_ptr<ET>(),
                                     arg0->get_shape(),
                                     arg1->get_shape(),
                                     broadcast_spec);
        return true;
    }

    bool evaluate_subtract(const HostTensorPtr& arg0,
                           const HostTensorPtr& arg1,
                           const HostTensorPtr& out,
                           const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1);
        switch (arg0->get_element_type())
        {
            TYPE_CASE(i8)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(i16)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(i32)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(i64)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(u8)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(u16)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(u32)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(u64)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(bf16)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(f16)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(f32)(arg0, arg1, out, broadcast_spec);
            break;
            TYPE_CASE(f64)(arg0, arg1, out, broadcast_spec);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Subtract::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_subtract(inputs[0], inputs[1], outputs[0], get_autob());
}

// ------------------------------- v1 ------------------------------------------

RTTI_DEFINITION("Subtract", op::v1::Subtract, Node, 1);

op::v1::Subtract::Subtract(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Subtract::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Subtract>(new_args.at(0), new_args.at(1), this->get_autob());
}

void op::v1::Subtract::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);

    adjoints.add_delta(x, delta);
    adjoints.add_delta(y, -delta);
}

bool op::v1::Subtract::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_subtract(inputs[0], inputs[1], outputs[0], get_autob());
}
