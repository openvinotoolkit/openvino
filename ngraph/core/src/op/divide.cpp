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

#include "ngraph/op/divide.hpp"
#include "ngraph/itt.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/divide.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ v0 -------------------------------------------

constexpr NodeTypeInfo op::v0::Divide::type_info;

op::v0::Divide::Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

op::v0::Divide::Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       bool pythondiv,
                       const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
    , m_pythondiv(pythondiv)
{
    constructor_validate_and_infer_types();
}

bool op::v0::Divide::visit_attributes(AttributeVisitor& visitor)
{
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    visitor.on_attribute("m_pythondiv", m_pythondiv);
    return true;
}

shared_ptr<Node> op::v0::Divide::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Divide>(
        new_args.at(0), new_args.at(1), this->is_pythondiv(), this->get_autob());
}

shared_ptr<Node> ngraph::operator/(const Output<Node>& arg0, const Output<Node>& arg1)
{
    return make_shared<op::v0::Divide>(arg0, arg1);
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec,
                  bool pythondiv)
    {
        runtime::reference::divide(arg0->get_data_ptr<ET>(),
                                   arg1->get_data_ptr<ET>(),
                                   out->get_data_ptr<ET>(),
                                   arg0->get_shape(),
                                   arg1->get_shape(),
                                   broadcast_spec,
                                   pythondiv);
        return true;
    }

    bool evaluate_divide(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& out,
                         const op::AutoBroadcastSpec& broadcast_spec,
                         bool pythondiv)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1);
        switch (arg0->get_element_type())
        {
            TYPE_CASE(i32)(arg0, arg1, out, broadcast_spec, pythondiv);
            break;
            TYPE_CASE(i64)(arg0, arg1, out, broadcast_spec, pythondiv);
            break;
            TYPE_CASE(u32)(arg0, arg1, out, broadcast_spec, pythondiv);
            break;
            TYPE_CASE(u64)(arg0, arg1, out, broadcast_spec, pythondiv);
            break;
            TYPE_CASE(f16)(arg0, arg1, out, broadcast_spec, pythondiv);
            break;
            TYPE_CASE(f32)(arg0, arg1, out, broadcast_spec, pythondiv);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Divide::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v0::Divide::evaluate");
    return evaluate_divide(inputs[0], inputs[1], outputs[0], get_autob(), is_pythondiv());
}

// ------------------------------ v1 -------------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::Divide, "Divide", 1, util::BinaryElementwiseArithmetic);

op::v1::Divide::Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

op::v1::Divide::Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       bool pythondiv,
                       const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
    , m_pythondiv(pythondiv)
{
    constructor_validate_and_infer_types();
}

bool op::v1::Divide::visit_attributes(AttributeVisitor& visitor)
{
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    visitor.on_attribute("m_pythondiv", m_pythondiv);
    return true;
}

shared_ptr<Node> op::v1::Divide::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Divide>(
        new_args.at(0), new_args.at(1), this->is_pythondiv(), this->get_autob());
}

bool op::v1::Divide::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v1::Divide::evaluate");
    return evaluate_divide(inputs[0], inputs[1], outputs[0], get_autob(), is_pythondiv());
}
