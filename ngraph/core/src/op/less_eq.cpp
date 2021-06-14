// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/less_eq.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/less_eq.hpp"

using namespace std;
using namespace ngraph;

// ---------------------------------- v1 ---------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::LessEqual, "LessEqual", 1);

op::v1::LessEqual::LessEqual(const Output<Node>& arg0,
                             const Output<Node>& arg1,
                             const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseComparison(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::LessEqual::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_LessEqual_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::LessEqual>(new_args.at(0), new_args.at(1), this->get_autob());
}

namespace less_equalop
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::less_eq(arg0->get_data_ptr<ET>(),
                                    arg1->get_data_ptr<ET>(),
                                    out->get_data_ptr<element::Type_t::boolean>(),
                                    arg0->get_shape(),
                                    arg1->get_shape(),
                                    broadcast_spec);
        return true;
    }

    bool evaluate_less_equal(const HostTensorPtr& arg0,
                             const HostTensorPtr& arg1,
                             const HostTensorPtr& out,
                             const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1, element::boolean);
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_less_equal, boolean, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_less_equal, i32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_less_equal, i64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_less_equal, u32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_less_equal, u64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_less_equal, f16, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_less_equal, f32, arg0, arg1, out, broadcast_spec);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace less_equalop

bool op::v1::LessEqual::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_LessEqual_evaluate);
    return less_equalop::evaluate_less_equal(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::LessEqual::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_LessEqual_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::boolean:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}
