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

#include "ngraph/op/hsigmoid.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/hsigmoid.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v5::HSigmoid, "HSigmoid", 5);

op::v5::HSigmoid::HSigmoid(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool op::v5::HSigmoid::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

shared_ptr<Node> op::v5::HSigmoid::clone_with_new_inputs(const OutputVector& new_args) const
{
    return make_shared<op::v5::HSigmoid>(new_args.at(0));
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;

        runtime::reference::hsigmoid<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_hsigmoid(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg);

        switch (arg->get_element_type())
        {
            TYPE_CASE(bf16)(arg, out, count);
            break;
            TYPE_CASE(f16)(arg, out, count);
            break;
            TYPE_CASE(f32)(arg, out, count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v5::HSigmoid::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    return evaluate_hsigmoid(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}
