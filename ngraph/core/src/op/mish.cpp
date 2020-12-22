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

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/mish.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/mish.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v4::Mish::type_info;

op::v4::Mish::Mish(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

bool op::v4::Mish::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v4::Mish::validate_and_infer_types()
{
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v4::Mish::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Mish>(new_args.at(0));
}

namespace mish
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mish<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_mish(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_mish, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_mish, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v4::Mish::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v4_Mish_evaluate)
    {
        return mish::evaluate_mish(inputs[0], outputs[0], shape_size(get_output_shape(0)));
    }
    return false;
}
