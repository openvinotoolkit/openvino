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

#include "itt.hpp"

#include "ngraph/op/exp.hpp"
#include "ngraph/op/multiply.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/exp.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Exp::type_info;

op::Exp::Exp(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Exp::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

shared_ptr<Node> op::Exp::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Exp>(new_args.at(0));
}

namespace expop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::exp<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_exp(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_exp, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_exp, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::Exp::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(
        v0_Exp_evaluate,
        return expop::evaluate_exp(inputs[0], outputs[0], shape_size(get_output_shape(0))));
    return false;
}
