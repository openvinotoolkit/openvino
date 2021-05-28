// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/subtract.hpp"
#include "itt.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/subtract.hpp"

using namespace std;
using namespace ngraph;

namespace subtract
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
            NGRAPH_TYPE_CASE(evaluate_subtract, i32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_subtract, i64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_subtract, u32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_subtract, u64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_subtract, f16, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_subtract, f32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_subtract, bf16, arg0, arg1, out, broadcast_spec);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace subtract

// ------------------------------- v1 ------------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::Subtract, "Subtract", 1, util::BinaryElementwiseArithmetic);

op::v1::Subtract::Subtract(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Subtract::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Subtract_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Subtract>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v1::Subtract::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Subtract_evaluate);
    return subtract::evaluate_subtract(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::Subtract::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Subtract_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::bf16: return true;
    default: break;
    }
    return false;
}
