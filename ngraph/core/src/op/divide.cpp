// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/divide.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/divide.hpp"

using namespace std;
using namespace ngraph;

namespace divide
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
            NGRAPH_TYPE_CASE(evaluate_divide, i32, arg0, arg1, out, broadcast_spec, pythondiv);
            NGRAPH_TYPE_CASE(evaluate_divide, i64, arg0, arg1, out, broadcast_spec, pythondiv);
            NGRAPH_TYPE_CASE(evaluate_divide, u32, arg0, arg1, out, broadcast_spec, pythondiv);
            NGRAPH_TYPE_CASE(evaluate_divide, u64, arg0, arg1, out, broadcast_spec, pythondiv);
            NGRAPH_TYPE_CASE(evaluate_divide, f16, arg0, arg1, out, broadcast_spec, pythondiv);
            NGRAPH_TYPE_CASE(evaluate_divide, f32, arg0, arg1, out, broadcast_spec, pythondiv);
            NGRAPH_TYPE_CASE(evaluate_divide, bf16, arg0, arg1, out, broadcast_spec, pythondiv);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace divide

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
    NGRAPH_OP_SCOPE(v1_Divide_visit_attributes);
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    visitor.on_attribute("m_pythondiv", m_pythondiv);
    return true;
}

shared_ptr<Node> op::v1::Divide::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Divide_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Divide>(
        new_args.at(0), new_args.at(1), this->is_pythondiv(), this->get_autob());
}

bool op::v1::Divide::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Divide_evaluate);
    return divide::evaluate_divide(inputs[0], inputs[1], outputs[0], get_autob(), is_pythondiv());
}

bool op::v1::Divide::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Divide_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::bf16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}
