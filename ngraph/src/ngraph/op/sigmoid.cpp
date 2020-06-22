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

#include "ngraph/op/sigmoid.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/sigmoid.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Sigmoid::type_info;
constexpr NodeTypeInfo op::SigmoidBackprop::type_info;

shared_ptr<Node> op::Sigmoid::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Sigmoid>(new_args.at(0));
}

op::Sigmoid::Sigmoid(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::sigmoid<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_sigmoid(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            TYPE_CASE(boolean)(arg0, out, count);
            break;
            TYPE_CASE(i32)(arg0, out, count);
            break;
            TYPE_CASE(i64)(arg0, out, count);
            break;
            TYPE_CASE(u32)(arg0, out, count);
            break;
            TYPE_CASE(u64)(arg0, out, count);
            break;
            TYPE_CASE(f16)(arg0, out, count);
            break;
            TYPE_CASE(f32)(arg0, out, count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::Sigmoid::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_sigmoid(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

op::SigmoidBackprop::SigmoidBackprop(const Output<Node>& arg, const Output<Node>& delta)
    : BinaryElementwiseArithmetic(arg, delta, AutoBroadcastSpec::NONE)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::SigmoidBackprop::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SigmoidBackprop>(new_args.at(0), new_args.at(1));
}

void op::Sigmoid::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto backprop = make_shared<op::SigmoidBackprop>(input_value(0), delta);
    adjoints.add_delta(input_value(0), backprop);
}
