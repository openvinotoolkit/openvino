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

#include "ngraph/op/swish.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/swish.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v4::Swish::type_info;

op::v4::Swish::Swish(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

op::v4::Swish::Swish(const Output<Node>& arg, const Output<Node>& beta)
    : Op({arg, beta})
{
    constructor_validate_and_infer_types();
}

bool op::v4::Swish::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v4::Swish::validate_and_infer_types()
{
    auto inputs_count = input_values().size();
    NODE_VALIDATION_CHECK(this,
                          inputs_count == 1 || inputs_count == 2,
                          "Swish must have 1 or 2 inputs, but it has: ",
                          inputs_count);

    if (inputs_count == 2)
    {
        NODE_VALIDATION_CHECK(this,
                              input_value(0).get_element_type() ==
                                  input_value(1).get_element_type(),
                              "Swish inputs must have the same type but they are: ",
                              input_value(0).get_element_type(),
                              " and ",
                              input_value(1).get_element_type());
        if (get_input_partial_shape(1).rank().is_static())
        {
            auto beta_rank = get_input_partial_shape(1).rank().get_length();
            NODE_VALIDATION_CHECK(this,
                                  beta_rank == 0,
                                  "Swish input with beta must be scalar but it has rank: ",
                                  beta_rank);
        }
    }
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v4::Swish::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() == 1)
    {
        return make_shared<op::v4::Swish>(new_args.at(0));
    }
    else
    {
        return make_shared<op::v4::Swish>(new_args.at(0), new_args.at(1));
    }
}

namespace swish
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& out,
                         const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        if (arg1 != nullptr)
        {
            runtime::reference::swish<T>(
                arg0->get_data_ptr<ET>(), arg1->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        }
        else
        {
            runtime::reference::swish<T>(
                arg0->get_data_ptr<ET>(), nullptr, out->get_data_ptr<ET>(), count);
        }
        return true;
    }

    bool evaluate_swish(const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out,
                        const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            TYPE_CASE(f16)(arg0, arg1, out, count);
            break;
            TYPE_CASE(f32)(arg0, arg1, out, count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v4::Swish::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    if (inputs.size() == 2)
    {
        return swish::evaluate_swish(
            inputs[0], inputs[1], outputs[0], shape_size(get_output_shape(0)));
    }
    else
    {
        return swish::evaluate_swish(
            inputs[0], nullptr, outputs[0], shape_size(get_output_shape(0)));
    }
}
