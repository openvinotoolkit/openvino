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

#include "ngraph/op/hswish.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/hswish.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v4::HSwish, "HSwish", 4);

op::v4::HSwish::HSwish(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool op::v4::HSwish::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

shared_ptr<Node> op::v4::HSwish::clone_with_new_inputs(const OutputVector& new_args) const
{
    return make_shared<op::v4::HSwish>(new_args.at(0));
}

namespace hswish
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;

        runtime::reference::hswish<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_hswish(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg);

        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_hswish, bf16, arg, out, count);
            NGRAPH_TYPE_CASE(evaluate_hswish, f16, arg, out, count);
            NGRAPH_TYPE_CASE(evaluate_hswish, f32, arg, out, count);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v4::HSwish::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v4_HSwish_evaluate)
    {
        return hswish::evaluate_hswish(inputs[0], outputs[0], shape_size(get_output_shape(0)));
    }
    return false;
}
