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

#include "ngraph/op/softplus.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/softplus.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v4::SoftPlus, "SoftPlus", 4);

op::v4::SoftPlus::SoftPlus(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

bool op::v4::SoftPlus::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v4::SoftPlus::validate_and_infer_types()
{
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v4::SoftPlus::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v4::SoftPlus>(new_args.at(0));
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::softplus<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_softplus(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count)
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

bool op::v4::SoftPlus::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::SoftPlus::evaluate");
    return evaluate_softplus(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}
